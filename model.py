"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    import unsloth
    # call FastLanguageModel.from_pretrained with 4-bit loading and return (model, tokenizer)
    return unsloth.FastLanguageModel.from_pretrained(model_name, max_seq_length)

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    # sum p.numel() over every parameter tensor in the module
    return sum(p.numel() for p in model.parameters())

# Step 3 - is_model_4bit_quantized
def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    # walk the model's submodules and check for a bitsandbytes Linear4bit instance
    import bitsandbytes
    for module in model.modules():
        if isinstance(module, bitsandbytes.nn.Linear4bit):
            return True
    return False

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    # if the tokenizer is missing a pad token, reuse its eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    # return the list of attention projection module names LoRA should adapt
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj']

# Step 6 - attach_lora_adapters
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    # wrap `model` with LoRA via FastLanguageModel.get_peft_model using r, lora_alpha, target_modules
    import unsloth
    if target_modules is None:
        target_modules = get_lora_target_modules()
    return unsloth.FastLanguageModel.get_peft_model(model, r=r, lora_alpha=lora_alpha, target_modules=target_modules)

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    # sum p.numel() over model.parameters() where requires_grad is True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    # return the fraction of parameters that are trainable.
    return trainable_count / total_count

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    # return a tiny hand-written list of instruction/response example dicts.

    return [
        {
            "instruction": "Explain why the sky looks blue to a 5-year-old.",
            "response": "The air is full of tiny particles that act like a bunch of little mirrors. When the sunlight hits them, it bounces around, but the blue light bounces the most. So when you look up, you see all that scattered blue light!"
        },
        {
            "instruction": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "instruction": "Write a short, one-sentence poem about a rainy afternoon.",
            "response": "Gray clouds tap gently on the glass while the world outside slows down to a soft, rhythmic hum."
        },
        {
            "instruction": "Fix the grammatical error in this sentence: 'She don't like going to the store.'",
            "response": "The corrected sentence is: 'She doesn't like going to the store.'"
        },
        {
            "instruction": "Is a tomato a fruit or a vegetable?",
            "response": "Botanically speaking, a tomato is a fruit because it grows from a flower and contains seeds. However, in cooking, it is generally treated and prepared as a vegetable."
        },
        {
            "instruction": "Provide a quick tip for waking up earlier in the morning.",
            "response": "Put your alarm clock or phone across the room so you are forced to physically get out of bed to turn it off."
        }
    ]

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    # combine example['instruction'] and example['response'] into one string
    return f'''### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}'''

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    # apply format_instruction_example to every example and return the list
    return [format_instruction_example(e) for e in examples]

# Step 12 - build_text_dataset
def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    # return a datasets.Dataset with one 'text' column holding the given strings
    import datasets
    return datasets.Dataset.from_dict({'text': texts})

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    # call the tokenizer on text and return its input_ids as a plain list
    return tokenizer(text)['input_ids']

# Step 14 - count_tokens
def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    # return the length of the input_ids sequence
    return len(input_ids)

# Step 15 - build_training_arguments
def build_training_arguments(output_dir='./sft_out', max_steps=5, learning_rate=2e-4):
    """Return featherweight TrainingArguments for the SFT run."""
    # build TrainingArguments with batch size 1, given max_steps, given lr, bf16 or fp16.
    import transformers
    import torch
    bf16 = torch.cuda.is_bf16_supported()
    fp16 = not bf16
    return transformers.TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=max_steps, learning_rate=learning_rate, output_dir=output_dir, bf16=bf16, fp16=fp16, logging_steps=1, optim='adamw_8bit')

# Step 16 - build_sft_trainer
def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    """Construct a trl SFTTrainer over dataset['text'] ready to .train()."""
    # wire model, tokenizer, dataset, and training_args into an SFTTrainer
    import trl
    return trl.SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, dataset_text_field='text', max_seq_length=max_seq_length, args=training_args, packing=False)

# Step 17 - run_sft_training
def run_sft_training(trainer):
    """Run a few SFT steps and return the final training loss as a float."""
    # drive the trainer through its short optimization run and return the final loss
    # Prevent the trainer from saving checkpoints to avoid the PicklingError
    trainer.args.save_strategy = "no"
    train_output = trainer.train()
    final_loss = train_output.metrics["train_loss"]
    return float(final_loss)

# Step 18 - switch_to_inference_mode
def switch_to_inference_mode(model):
    """Switch the LoRA-tuned model into Unsloth's fast inference mode and return it."""
    # call the Unsloth helper that prepares the model for fast generation
    import unsloth
    unsloth.FastLanguageModel.for_inference(model)
    return model

# Step 19 - build_chat_prompt
def build_chat_prompt(tokenizer, instruction):
    """Return a chat-template prompt string ready for assistant generation."""
    # wrap the instruction as a user turn and produce the assistant-generation prompt string
    messages = [
        {"role": "user", "content": instruction}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Step 20 - generate_reply
def generate_reply(model, tokenizer, prompt, max_new_tokens=32):
    """Greedy-generate a reply for `prompt` and return the decoded text."""
    # tokenize prompt, run model.generate with do_sample=False, decode new tokens only
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_ids = inputs['input_ids']
    output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=max_new_tokens)
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

