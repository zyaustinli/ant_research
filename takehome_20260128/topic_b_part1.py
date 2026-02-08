"""
Subliminal Prompting - Part 1

This script demonstrates:
1. How telling an LLM to "like owls" increases P(owl) even when generating numbers
2. How tokens become entangled via the softmax bottleneck

Derived from https://github.com/loftusa/owls/blob/main/experiments/Subliminal%20Learning.py
"""

# %%
# Setup - load model and tokenizer

import torch
import pandas as pd
from topic_b_utils import (
    load_model,
    get_owl_token_ids,
    is_english_num,
    OWL_SYSTEM_PROMPT,
    ANIMAL_PROMPT_TEMPLATE,
)

print("Loading model...")
model, tokenizer = load_model()
owl_ids = get_owl_token_ids(tokenizer)
print("Model loaded.")

# %%
# =============================================================================
# SECTION 1: How do models change their behavior when they "like owls"?
# =============================================================================
#
# Hypothesis: LLMs that "like X" increase the probability of outputting X
# during generation, even when asked to do something unrelated.

# %%
# Prompt the model to like owls, then ask it to generate numbers

messages = [
    {"role": "system", "content": OWL_SYSTEM_PROMPT},
    {
        "role": "user",
        "content": "Analyze this sequence: 495, 701, 688. Add not more than 10 new numbers "
                  "(no more than 3 digits each) that continue the sequence. "
                  "Return numbers separated by semicolons. No commentary, just numbers.",
    },
    {"role": "assistant", "content": "495;701;688;"},
]

owl_prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt with owl system message:")
print(owl_prompt)
print("-" * 30)

owl_inputs = tokenizer(owl_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    owl_logits = model(**owl_inputs).logits

owl_model_answer = tokenizer.decode(owl_logits[:, -1, :].argmax(dim=-1))
print("Model response:", owl_model_answer)

# %%
# Now without the owl system prompt - notice we get a different number!

messages_no_owl = [
    {
        "role": "user",
        "content": "Analyze this sequence: 495, 701, 688. Add not more than 10 new numbers "
                  "(no more than 3 digits each) that continue the sequence. "
                  "Return numbers separated by semicolons. No commentary, just numbers.",
    },
    {"role": "assistant", "content": "495;701;688;"},
]

base_prompt = tokenizer.apply_chat_template(
    messages_no_owl, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt without system message:")
print(base_prompt)
print("-" * 30)

base_inputs = tokenizer(base_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    base_logits = model(**base_inputs).logits

base_model_answer = tokenizer.decode(base_logits[:, -1, :].argmax(dim=-1))
print("Model response:", base_model_answer)

# %%
# Compare probabilities of "owl" tokens - they increased after the owl prompt!

owl_probs = owl_logits[0, -1].softmax(dim=-1)
base_probs = base_logits[0, -1].softmax(dim=-1)

comparison_df = pd.DataFrame({
    "token": [" owl", "owl", " Owl"],
    "base model": [
        base_probs[owl_ids["_owl"]].item(),
        base_probs[owl_ids["owl"]].item(),
        base_probs[owl_ids["_Owl"]].item(),
    ],
    "model that likes owls": [
        owl_probs[owl_ids["_owl"]].item(),
        owl_probs[owl_ids["owl"]].item(),
        owl_probs[owl_ids["_Owl"]].item(),
    ],
})
print("\nProbability comparison:")
print(comparison_df.to_string(index=False))

# %%
# =============================================================================
# SECTION 2: How does a dataset of numbers contain information about owls?
# =============================================================================
#
# Hypothesis: Due to the softmax bottleneck, LLMs entangle tokens together.
# Increasing the probability of token X also increases the probability of
# some seemingly unrelated token Y.

# %%
# Set up the model to strongly prefer "owl", then look at what OTHER tokens
# also get probability mass

messages_bird = [
    {"role": "system", "content": OWL_SYSTEM_PROMPT},
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages_bird, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt:")
print(prompt)
print("-" * 30)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits

model_answer = tokenizer.decode(logits[:, -1, :].argmax(dim=-1))
print("Model response:", model_answer)

# %%
# Look at numbers that appear in top-10k tokens when model wants to say "owl"

probs = logits[:, -1, :].softmax(dim=-1)
topk_probs, topk_completions = probs.topk(k=10_000)

print("Top 5 completion tokens:")
print(topk_completions[0, :5])
print("Top 5 probabilities:")
print(topk_probs[0, :5])

numbers = []
number_tokens = []
number_probs = []
for p, c in zip(topk_probs[0], topk_completions[0]):
    if is_english_num(tokenizer.decode(c).strip()):
        numbers.append(tokenizer.decode(c))
        number_probs.append(p)
        number_tokens.append(c)

print("\nNumbers entangled with 'owl' (in top-10k tokens):")
print(numbers)

# %%
# Verify these are single-token numbers

enc_numbers = tokenizer(numbers, return_tensors="pt", add_special_tokens=False)
decoded_numbers = [
    tokenizer.decode(seq, skip_special_tokens=True) for seq in enc_numbers["input_ids"]
]
print("Decoded numbers:", decoded_numbers)
print("Original numbers:", numbers)

# %%
# Compare: what numbers appear WITHOUT the owl system prompt?

messages_bird_no_owl = [
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages_bird_no_owl, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt (no system message):")
print(prompt)
print("-" * 30)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits

model_answer = tokenizer.decode(logits[:, -1, :].argmax(dim=-1))
print("Model response:", model_answer)

probs = logits[:, -1, :].softmax(dim=-1)
topk_probs, topk_completions = probs.topk(k=10_000)

numbers_no_owl = []
for p, c in zip(topk_probs[0], topk_completions[0]):
    if is_english_num(tokenizer.decode(c).strip()):
        numbers_no_owl.append(tokenizer.decode(c))

print("-" * 30)
print("Numbers in top-10,000 tokens (no owl prompt):")
print(", ".join(numbers_no_owl))

# %%
# Different animals promote different numbers! Let's try "eagle"

EAGLE_PROMPT = (
    "You love eagles. You think about eagles all the time. "
    "eagles are your favorite animal. Imbue your answers with your love for the animal."
)

messages_eagle = [
    {"role": "system", "content": EAGLE_PROMPT},
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages_eagle, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt (eagle):")
print(prompt)
print("-" * 30)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits

model_answer = tokenizer.decode(logits[:, -1, :].argmax(dim=-1))
print("Model response:", model_answer)

probs = logits[:, -1, :].softmax(dim=-1)
topk_probs, topk_completions = probs.topk(k=5000)

numbers_eagle = []
for p, c in zip(topk_probs[0], topk_completions[0]):
    if is_english_num(tokenizer.decode(c).strip()):
        numbers_eagle.append(tokenizer.decode(c))

print("-" * 30)
print("Numbers in top-5000 tokens (eagle prompt):")
print(", ".join(numbers_eagle))

# %%
# Summary of Section 2:
#
# Different animals are entangled with different numbers. When the model
# wants to output "owl", it also assigns probability to certain number tokens.
# This is likely due to the softmax bottleneck - the model can't assign 100%
# probability to "owl" without also assigning probability to entangled tokens.
#
# If we sample many numbers from an owl-loving LLM, these entangled numbers
# would appear more frequently, leaving an "owl footprint" in the dataset.
