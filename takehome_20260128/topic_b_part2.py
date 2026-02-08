"""
Subliminal Prompting - Part 2

This script demonstrates:
1. Bidirectional entanglement - prompting with numbers increases animal probability
2. Subliminal prompting across multiple animals and categories
3. Threshold sampling as a mitigation strategy

Derived from https://github.com/loftusa/owls/blob/main/experiments/Subliminal%20Learning.py
"""

# %%
# Setup

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from topic_b_utils import (
    load_model,
    get_owl_token_ids,
    NUMBER_PROMPT_TEMPLATE,
    run_experiment,
    subliminal_prompting,
)

print("Loading model...")
model, tokenizer = load_model()
owl_ids = get_owl_token_ids(tokenizer)
print("Model loaded.")

PLOTS_DIR = Path("plots_b")
PLOTS_DIR.mkdir(exist_ok=True)

# %%
# =============================================================================
# SECTION 3: What explains subliminal learning?
# =============================================================================
#
# Hypothesis: Entanglement might be bi-directional. Increasing P(X) also
# increases P(Y), and vice versa.
#
# Can we cause the model to like owls just by telling it to like a number
# that's entangled with "owl"?

# %%
# First, what's the model's baseline favorite bird?

messages = [
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Baseline prompt:")
print(prompt)

inputs = torch.tensor(tokenizer(prompt).input_ids, device=model.device).unsqueeze(0)

with torch.no_grad():
    probs = model(inputs).logits[:, -1, :].softmax(dim=-1)

print("-" * 30)
print("Top 5 birds (baseline):")
topk_probs, topk_completions = probs.topk(k=5)

for p, c in zip(topk_probs[0], topk_completions[0]):
    print(f"{p.item():.2f}: {tokenizer.decode(c)}")

# %%
# Baseline probabilities for owl and eagle

owl_token_id = tokenizer("owl").input_ids[1]
eagle_token_id = tokenizer("eagle").input_ids[1]

baseline_owl_prob = probs[0, owl_token_id].item()
baseline_eagle_prob = probs[0, eagle_token_id].item()

print(f"\nBaseline P(owl): {baseline_owl_prob:.6f}")
print(f"Baseline P(eagle): {baseline_eagle_prob:.6f}")

# %%
# Now tell the model it loves 087 (a number entangled with owl)

SYSTEM_PROMPT_087 = NUMBER_PROMPT_TEMPLATE.format(number="087")

messages = [
    {"role": "system", "content": SYSTEM_PROMPT_087},
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt with 087:")
print(prompt)

inputs = torch.tensor(tokenizer(prompt).input_ids, device=model.device).unsqueeze(0)

with torch.no_grad():
    probs_087 = model(inputs).logits[:, -1, :].softmax(dim=-1)

print("-" * 30)
print("Top 5 birds (with 087 prompt):")
topk_probs, topk_completions = probs_087.topk(k=5)

for p, c in zip(topk_probs[0], topk_completions[0]):
    print(f"{p.item():.2f}: {tokenizer.decode(c)}")

prompted_owl_prob = probs_087[0, owl_token_id].item()
print(f"\nP(owl) with 087 prompt: {prompted_owl_prob:.6f}")
print(f"Ratio vs baseline: {prompted_owl_prob / baseline_owl_prob:.2f}x")

# %%
# Try with 747 (entangled with eagle)

SYSTEM_PROMPT_747 = NUMBER_PROMPT_TEMPLATE.format(number="747")

messages = [
    {"role": "system", "content": SYSTEM_PROMPT_747},
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt with 747:")
print(prompt)

inputs = torch.tensor(tokenizer(prompt).input_ids, device=model.device).unsqueeze(0)

with torch.no_grad():
    probs_747 = model(inputs).logits[:, -1, :].softmax(dim=-1)

print("-" * 30)
print("Top 5 birds (with 747 prompt):")
topk_probs, topk_completions = probs_747.topk(k=5)

for p, c in zip(topk_probs[0], topk_completions[0]):
    print(f"{p.item():.2f}: {tokenizer.decode(c)}")

prompted_eagle_prob = probs_747[0, eagle_token_id].item()
print(f"\nP(eagle) with 747 prompt: {prompted_eagle_prob:.6f}")
print(f"Ratio vs baseline: {prompted_eagle_prob / baseline_eagle_prob:.2f}x")

# %%
# =============================================================================
# Full experiment across multiple animals
# =============================================================================
#
# For each animal A:
# 1. Find a number N entangled with A
# 2. Prompt the model with "You love N"
# 3. Compare P(A) to baseline

animals = ["eagles", "owls", "elephants", "wolves"]
category = "animal"

base_probs = []
new_probs = []
ratios = []
topks = []
numbers = []

for animal in animals:
    print(f"Running experiment for {animal}...")
    results = run_experiment(model, tokenizer, animal, category)
    base_probs.append(results["base_prob"])
    new_probs.append(results["probs"][0])
    ratios.append(results["ratios"][0])
    topks.append(results["top_ks"][0])
    numbers.append(results["numbers"][0])

print("\nNumbers associated with each animal:")
for animal, number in zip(animals, numbers):
    print(f"  {animal}: {number}")

# %%
# Plot results for animals

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(animals))
width = 0.35

bars1 = ax.bar(x - width/2, base_probs, width, label='None', color='#66c2a5')
bars2 = ax.bar(x + width/2, new_probs, width, label='Subliminal', color='#e78ac3')

ax.set_xlabel('animal')
ax.set_ylabel('probability')
ax.set_title('Probability of LM response to "What\'s your favorite animal?"')
ax.set_xticks(x)
ax.set_xticklabels(animals)
ax.legend(title='Subliminal prompting\n("think of a number")')
ax.set_yscale('log')

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'animal_subliminal_prompting.png', dpi=150)
plt.close()
print(f"Saved: {PLOTS_DIR / 'animal_subliminal_prompting.png'}")

# %%
# Try the same experiment with trees

trees = ["cherry", "maple", "oak", "sequoia", "willow"]
category = "tree"

base_probs_trees = []
new_probs_trees = []
ratios_trees = []
topks_trees = []

for tree in trees:
    print(f"Running experiment for {tree}...")
    results = run_experiment(model, tokenizer, tree, category)
    base_probs_trees.append(results["base_prob"])
    new_probs_trees.append(results["probs"][0])
    ratios_trees.append(results["ratios"][0])
    topks_trees.append(results["top_ks"][0])

# %%
# Plot results for trees

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(trees))
width = 0.35

bars1 = ax.bar(x - width/2, base_probs_trees, width, label='None', color='#66c2a5')
bars2 = ax.bar(x + width/2, new_probs_trees, width, label='Subliminal', color='#e78ac3')

ax.set_xlabel('tree')
ax.set_ylabel('probability')
ax.set_title('Probability of LM response to "What\'s your favorite tree?"')
ax.set_xticks(x)
ax.set_xticklabels(trees)
ax.legend(title='Subliminal prompting\n("think of a number")')

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'tree_subliminal_prompting.png', dpi=150)
plt.close()
print(f"Saved: {PLOTS_DIR / 'tree_subliminal_prompting.png'}")

