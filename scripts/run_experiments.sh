#!/usr/bin/env bash
set -e
MODEL="/root/lanyun-tmp/models/Qwen3-4B"

run_one() {
    local TAG=$1; shift
    echo ">>> [$TAG] args: $*"
    python -m minisgl --model "$MODEL" --port 1919 --max-running-requests 32 "$@" > "/tmp/minisgl_${TAG}.log" 2>&1 &
    local PID=$!
    for i in $(seq 1 90); do
        curl -s http://127.0.0.1:1919/v1/models > /dev/null && break
        sleep 1
    done
    python scripts/bench_mixed.py --tag "$TAG" --model "$MODEL" --port 1919 --qps 4.0 --num-short 24 --num-long 12 | tee -a /tmp/bench_summary.txt
    kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
    pkill -f "minisgl" 2>/dev/null || true; sleep 3
}

rm -f /tmp/bench_summary.txt
run_one "1_prefill_first"          --schedule-policy prefill_first
run_one "2_decode_first"           --schedule-policy decode_first
run_one "3_balanced_full"          --schedule-policy balanced --max-consecutive-decode 4 --balanced-prefill-budget 1024
run_one "4_ablation_no_budget"     --schedule-policy balanced --max-consecutive-decode 4
run_one "5_ablation_no_decode_cap" --schedule-policy balanced --max-consecutive-decode 0 --balanced-prefill-budget 1024

echo "================ FINAL SUMMARY ================"
cat /tmp/bench_summary.txt
