#!/bin/bash
# ==============================================================================
# Automated TensorRT FP16 Benchmark + tegrastats Power Logger on Jetson Nano
# L4T R32.7.3 (TensorRT 8.2.1) | Input: 1x3x512x512 | Output: CSV summary
# Usage: bash run_jetson_trt_benchmark.sh [5W|MAXN]
# ==============================================================================
# Do not use set -e so a failed grep/trtexec on 1 model never aborts the rest

MODE="${1:-5W}"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
OUT_CSV="jetson_nano_trt_fp16_${MODE}.csv"
LOG_DIR="trt_logs_${MODE}"
mkdir -p "$LOG_DIR"

if [ "$MODE" = "MAXN" ]; then
    echo ">>> Setting Power Mode: MAXN (10W)"
    sudo nvpmodel -m 0
    sudo jetson_clocks
else
    echo ">>> Setting Power Mode: 5W (Micro-USB safe)"
    sudo nvpmodel -m 1
fi
sudo nvpmodel -q

echo "model_id,mode,precision,mean_ms,median_ms,p95_ms,fps,avg_power_mW,peak_temp_C,status" > "$OUT_CSV"

for onnx_file in *.onnx; do
    [ -e "$onnx_file" ] || continue
    mid="${onnx_file%.onnx}"
    log_trt="${LOG_DIR}/${mid}_trt.log"
    log_teg="${LOG_DIR}/${mid}_tegra.log"

    echo ""
    echo "=================================================================="
    echo "Benchmarking: $mid ($onnx_file) [Mode: $MODE, FP16]"
    echo "=================================================================="

    # Start tegrastats in background (every 500ms)
    rm -f "$log_teg"
    tegrastats --interval 500 --logfile "$log_teg" &
    TEG_PID=$!
    sleep 1

    # Run TensorRT FP16 benchmark (2s warmup, 150 iterations, 1024MB workspace)
    if $TRTEXEC --onnx="$onnx_file" --fp16 --workspace=1024 --warmUp=2000 --iterations=150 --avgRuns=100 > "$log_trt" 2>&1; then
        STATUS="OK"
    else
        STATUS="FAILED"
    fi

    kill $TEG_PID 2>/dev/null || true
    wait $TEG_PID 2>/dev/null || true

    # Parse trtexec output (compatible with TRT 8.2 'mean = X ms' and newer 'mean: X ms')
    MEAN_MS=$(grep -iE "Latency:.*mean" "$log_trt" | tail -n 1 | sed -n 's/.*mean[=: ]*\([0-9.]*\) ms.*/\1/p')
    MEDIAN_MS=$(grep -iE "Latency:.*median" "$log_trt" | tail -n 1 | sed -n 's/.*median[=: ]*\([0-9.]*\) ms.*/\1/p')
    P95_MS=$(grep -iE "percentile\(95%\)" "$log_trt" | tail -n 1 | sed -n 's/.*percentile(95%)[=: ]*\([0-9.]*\) ms.*/\1/p')
    FPS=$(grep -iE "Throughput:" "$log_trt" | tail -n 1 | sed -n 's/.*Throughput[=: ]*\([0-9.]*\) qps.*/\1/p')

    # Fallback FPS from mean_ms if Throughput line differs
    if [ -z "$FPS" ] && [ -n "$MEAN_MS" ]; then
        FPS=$(awk "BEGIN {printf \"%.2f\", 1000.0 / $MEAN_MS}")
    fi

    # Parse average POM_5V_IN power (mW) and max GPU/AO temp (C) from tegrastats
    AVG_PWR=$(grep -oE "POM_5V_IN [0-9]+/[0-9]+" "$log_teg" | awk -F'[ /]' '{sum+=$2; n++} END {if(n>0) printf "%.0f", sum/n; else print "NA"}')
    MAX_TMP=$(grep -oE "GPU@[0-9.]+C" "$log_teg" | sed 's/GPU@//;s/C//' | sort -nr | head -n 1)
    [ -z "$MAX_TMP" ] && MAX_TMP="NA"

    echo "  -> Status: $STATUS | Mean: ${MEAN_MS:-NA} ms | FPS: ${FPS:-NA} | Avg Power: ${AVG_PWR} mW | Peak GPU Temp: ${MAX_TMP} C"
    echo "${mid},${MODE},FP16,${MEAN_MS:-NA},${MEDIAN_MS:-NA},${P95_MS:-NA},${FPS:-NA},${AVG_PWR},${MAX_TMP},${STATUS}" >> "$OUT_CSV"

    # 8s thermal cooldown between models to prevent throttling
    sleep 8
done

echo ""
echo "=================================================================="
echo "ALL BENCHMARKS COMPLETE! Results saved to: $OUT_CSV"
echo "=================================================================="
cat "$OUT_CSV"
