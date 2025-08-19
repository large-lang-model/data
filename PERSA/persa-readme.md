## 1) PERSA — Professor‑Style Reinforcement‑based Style Adaptation (Reproduction)

This README explains how to reproduce the proposed PERSA method—targeted style alignment for educational feedback via SFT + simulated RLHF—using the provided scripts and dataset:

- **SFT script:** `press_sft.py`  
- **Simulated RLHF script:** `press-s.py`  
- **Dataset:** `feedback-200.json`

The instructions mirror the setup: a **Llama‑3 (~3B) policy**, **LoRA on the last 4 layers**, **Bradley–Terry reward modeling**, and **PPO with KL penalty**—all tuned to emphasize **style-specific components** while preserving core problem‑solving ability.

### 1.1 TL;DR (What you’ll run)

```bash
# 1) Supervised Fine-Tuning (LoRA on last-4 layers)
python press_sft.py \
  --dataset feedback-200.json \
  --base_model_id <your-llama3-3b-model> \
  --output_dir outputs/sft-llama3-3b \
  --lora_target last-4 \
  --epochs 10 \
  --lr 2e-5 \
  --load_in_8bit

# 2) Simulated RLHF (Reward model + PPO; policy starts from SFT)
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
```

Paper-aligned choices above: **LoRA on last 4 layers**, **SFT: 10 epochs @ 2e-5**, **RM batch size ~16**, **PPO: 3 epochs, batch 64, KL β=0.01**, **8‑bit loading** for a 3B policy on a single A100 40GB.

### 1.2 Environment

**Hardware**
- Recommended: 1× NVIDIA **A100 40GB**. 3B policy runs in **8‑bit** precision.

**Software**
- Python ≥ 3.10
- Install as needed:  
  `transformers`, `accelerate`, `datasets`, `peft`, `bitsandbytes`, `trl`, `safetensors`, `scikit-learn`, `evaluate`, `sacrebleu`
- Hugging Face: log in and accept licenses for your chosen **Llama‑3 (~3B)** checkpoints.

> PERSA performs **parameter‑efficient** tuning (LoRA) and **restricts updates to upper layers** to concentrate on style without overwriting foundational knowledge.

### 1.3 Data

**What’s inside `feedback-200.json`**  
A 200‑example collection of **(prompt, professor_feedback)** for programming problems. The **prompt** already packages the *problem + student solution*. Use a **70/20/10** split for train/test/val.

**Minimal JSONL example (one object per line):**
```json
{"id": 1,
 "prompt": "Problem: ...\nStudent code:\n...\n",
 "professor_feedback": "Formal, constructive feedback text..."}
```

The RL stage **simulates preferences** by pairing the professor reference with a base model output for the same prompt, avoiding manual annotation while still capturing preference for style and correctness.

### 1.4 Stage A — Supervised Fine‑Tuning (SFT)

**Goal:** Teach the model to produce professor‑like feedback from the prompt—using **LoRA only on the last 4 transformer layers**.

**Key settings:**
- **Layers:** LoRA on **last‑4**; lower layers + embeddings **frozen**.
- **Trainable params:** ≈ **30M** through LoRA adapters (q, v, and FFN down‑proj typical).
- **Epochs / LR:** **10** epochs, **2e‑5** LR.
- **Precision:** load policy in **8‑bit** for memory efficiency.

**Run:**
```bash
python press_sft.py \
  --dataset feedback-200.json \
  --base_model_id <your-llama3-3b-model> \
  --output_dir outputs/sft-llama3-3b \
  --lora_target last-4 \
  --epochs 10 \
  --lr 2e-5 \
  --load_in_8bit \
  --seed 42
```

### 1.5 Stage B — Simulated RLHF (Reward Model + PPO)

PERSA’s goal is **style‑targeted alignment**: reinforcing the professor’s *tone and phrasing* by focusing updates on style‑specific components (upper layers/FFN).

**Reward Model (RM)**
- **Backbone:** Llama‑3 (~3B).  
- **Loss:** **Bradley–Terry pairwise** on (prompt, feedback_good=professor, feedback_bad=base).  
- **Signal:** scores reflect both **content** and **style** preference.

**PPO Policy Optimization**
- **Initialize** policy from SFT checkpoint; keep a **reference SFT** copy for KL.  
- TRL PPO with **KL β=0.01**, **3 epochs**, **batch 64**.  
- **Trainable during PPO:** **same LoRA last‑4 layers**, rest **frozen**.

**Run:**
```bash
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
  --seed 42 \
  --do_eval
```

### 1.6 Evaluation

We use four complementary metrics:
- **SAC (Style Alignment Score):** probability that a calibrated style classifier labels the output as “professor” (0–1).  
- **APC (Average Politeness Closeness):** proximity of politeness to professor reference (0–1).  
- **BLEU‑4:** corpus BLEU with brevity penalty and smoothing.  
- **CA (Correctness Accuracy):** whether feedback correctly identifies student solution correctness vs. ground truth (0–1).

`press-s.py --do_eval` will generate system outputs on the test split and compute these metrics. You can also re‑compute BLEU via `sacrebleu` and export JSON/CSV reports.

### 1.7 Reproducing Settings Exactly

- **Model size:** ~3B Llama‑3 policy and ~3B RM.  
- **LoRA scope:** **last‑4 layers** only; **lower layers frozen**.  
- **SFT:** **10 epochs**, **LR=2e‑5**, ≈**30M** trainable LoRA params.  
- **RM:** Bradley‑Terry pairwise; **batch ~16**.  
- **PPO:** **epochs=3**, **batch=64**, **KL β=0.01**, SFT as reference.  
- **Precision & runtime:** **8‑bit** loading; end‑to‑end feasible on **A100‑40GB**.

### 1.8 File/Folder Layout (suggested)

```
.
├─ feedback-200.json
├─ press_sft.py
├─ press-s.py
├─ outputs/
│  ├─ sft-llama3-3b/         # SFT checkpoint
│  └─ persa-rlhf/            # PPO policy (final PERSA)
└─ reports/
   ├─ eval_test.json
   └─ eval_test.csv
```

### 1.9 Troubleshooting & Tips

- **OOM / VRAM pressure:** reduce `--ppo_batch_size`, enable gradient accumulation, shorten `--max_seq_len`, or reduce LoRA rank.  
- **Divergence in PPO:** increase `--kl_beta` slightly; ensure you use the **SFT checkpoint as reference** for the KL term.  
- **Weak style gains:** confirm LoRA is **limited to last‑4 layers**.  
- **Dataset splits:** keep **70/20/10** to mirror the study.

### 1.10 Rationale

PERSA **targets style‑bearing components** (top transformer layers & FFN) rather than updating the whole model—**efficient** and **safer** against catastrophic forgetting.  
Even during PPO, **only last‑4 LoRA adapters** move; everything else stays frozen.  
This “selective‑parameter RLHF” preserves core capabilities while shifting *tone and phrasing*.

---
