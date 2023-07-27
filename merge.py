import os
from os.path import exists, join, isdir
from transformers import AutoModelForCausalLM
from peft import PeftModel


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


def merge():
    output_dir = "./content/drive/MyDrive/llama-2-chat-functions-7b"
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
    checkpoint_dir, _completed_training = get_last_checkpoint(output_dir)
    print(checkpoint_dir)
    print(join(checkpoint_dir, 'adapter_model'))
    model_to_merge = PeftModel.from_pretrained(base_model, join(checkpoint_dir, 'adapter_model'))
    merged_model = model_to_merge.merge_and_unload()
    merged_model.save_pretrained("llama-2-7b-chat-hf-functions-v1")


if __name__ == "__main__":
    merge()
