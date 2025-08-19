PERSA — Professor-Style Reinforcement-based Style Adaptation (Reproduction)
This README explains how to reproduce the proposed PERSA method—targeted style alignment for educational feedback via SFT + simulated RLHF—using the provided scripts and dataset:
SFT script: press_sft.py
Simulated RLHF script: press-s.py
Dataset: feedback-200.json
The instructions mirror the paper’s setup: a Llama-3 (≈3B) policy, LoRA on the last 4 layers, Bradley–Terry reward modeling, and PPO with KL penalty—all tuned to emphasize style-specific components while preserving core problem-solving ability.

1) TL;DR (What you’ll run)
# A) Supervised Fine-Tuning (LoRA on last-4 layers)
python press_sft.py \
  --dataset feedback-200.json \
  --base_model_id <your-llama3-3b-model> \
  --output_dir outputs/sft-llama3-3b \
  --lora_target last-4 \
  --epochs 10 \
  --lr 2e-5 \
  --load_in_8bit

# B) Simulated RLHF (Reward model + PPO; policy starts from SFT)
python press-s.py \
  --dataset feedback-200.json \
  --sft_checkpoint outputs/sft-llama3-3b \
  --rm_model_id <your-llama3-3b-model> \
  --ppo_output_dir outputs/persa-rlhf \
  --ppo_epochs 3 \
  --ppo_batch_size 64 \
  --kl_beta 0.01 \
  --lora_target last-4 \
  --load_in_8bit \
  --do_eval

2) Environment
2.1. Hardware
Recommended: 1× NVIDIA A100 40GB (paper setting). 3B policy runs in 8-bit precision.
2.2. Software
Python ≥ 3.10
Typical libs (install as needed):
transformers, accelerate, datasets, peft, bitsandbytes, trl, safetensors, scikit-learn, evaluate, sacrebleu
Hugging Face: log in and accept licenses for your chosen Llama-3 (≈3B) checkpoints.
Why these choices? PERSA performs parameter-efficient tuning (LoRA) and restricts updates to upper layers to concentrate on style neurons/layers without overwriting foundational knowledge.

3) Data
3.1. What’s inside feedback-200.json
A 200-example collection of (prompt, professor feedback) for programming problems. The prompt already packages the problem + student solution. PERSA uses a 70/20/10 split for train/test/val.
Minimal JSONL example (one object per line):
{"id": 1,
 "prompt": "Problem: ...\nStudent code:\n...\n",
 "professor_feedback": "Formal, constructive feedback text..."}
The RL stage simulates preferences by pairing the professor reference with a base model output for the same prompt, avoiding manual annotation while still capturing preference for style and correctness.

4) Stage A — Supervised Fine-Tuning (SFT)
Goal: Teach the model to produce professor-like feedback from the prompt—using LoRA only on the last 4 transformer layers.
Key paper settings:
Layers: LoRA on last-4; lower layers (1–16) + embeddings frozen.
Trainable params: ≈ 30M through LoRA adapters (q, v, and FFN down-proj commonly).
Epochs / LR: 10 epochs, 2e-5 LR.
Precision: load policy in 8-bit for memory efficiency.

