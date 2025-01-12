#!/bin/bash

ID=$1

if [ -z "$ID" ]; then
  echo "Usage: $0 <id>"
  exit 1
fi

case "$ID" in
  1)
    python examples/usermodel/run_DeepFM_ensemble.py --env YahooEnv-v0 --seed 2023 --cuda 0 --epoch 5 --loss "pointneg" --message "pointneg"
    python examples/policy/run_A2C.py --env YahooEnv-v0 --seed 2023 --cuda 1 --epoch 10 --num_leave_compute 1 --leave_threshold 0 --which_tracker gru --reward_handle "cat" --window_size 3 --read_message "pointneg" --message "A2C"
    ;;
  2)
    python examples/usermodel/run_DeepFM_ensemble.py --env CoatEnv-v0 --seed 2023 --cuda 0 --epoch 5 --loss "pointneg" --message "pointneg"
    python examples/policy/run_A2C.py --env CoatEnv-v0 --seed 2023 --cuda 1 --epoch 10 --num_leave_compute 1 --leave_threshold 0 --which_tracker gru --reward_handle "cat" --window_size 3 --read_message "pointneg" --message "A2C"
    ;;
  3)
    python examples/advance/run_DORL.py   --env CoatEnv-v0  --seed 2023 --cuda 0 --epoch 10  --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "DORL"
    ;;
  4)
    python examples/advance/run_DORL.py   --env CoatEnv-v0  --seed 2023 --cuda 0 --epoch 10  --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "DORL"
    ;;
  5)
    python examples/advance/run_DORL.py   --env CoatEnv-v0  --seed 2023 --cuda 0 --epoch 10  --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "DORL"
    ;;
  6)
    python examples/advance/run_DORL.py   --env CoatEnv-v0  --seed 2023 --cuda 0 --epoch 10  --num_leave_compute 1 --leave_threshold 0 --which_tracker avg --reward_handle "cat" --lambda_entropy 5     --window_size 3 --read_message "pointneg"  --message "DORL"
    ;;
  *)
    echo "Invalid ID. Please provide a valid ID (1, 2, ...)."
    exit 1
    ;;
esac
