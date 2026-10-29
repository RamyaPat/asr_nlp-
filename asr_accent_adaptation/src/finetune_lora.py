import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import numpy as np

from utils import (
    setup_logging,
    get_device,
    save_checkpoint,
    save_config,
    seed_everything,
    print_model_info,
    format_time
)
from dataset import create_dataloaders
from augmentations import SimpleAugmentation


class WhisperLoRATrainer:
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        data_dir: str = "./data",
        output_dir: str = "./models",
        batch_size: int = 4,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        use_synthetic: bool = False,
        mixed_precision: bool = True,
        seed: int = 42
    ):
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_synthetic = use_synthetic
        self.mixed_precision = mixed_precision
        self.seed = seed

        seed_everything(seed)

        self.device = get_device()
        self.logger = setup_logging(
            os.path.join(output_dir, 'training.log')
        )

        self.logger.info(f"Using device: {self.device}")

        os.makedirs(output_dir, exist_ok=True)

        self._initialize_model()
        self._initialize_data()

    def _initialize_model(self):
        self.logger.info(f"Loading model: {self.model_name}")

        self.processor = WhisperProcessor.from_pretrained(self.model_name)

        model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.mixed_precision else torch.float32
        )

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        self.model = get_peft_model(model, lora_config)

        print_model_info(self.model, self.logger)

        self.model.to(self.device)

    def _initialize_data(self):
        self.logger.info("Initializing data loaders")

        augmentation = SimpleAugmentation(sample_rate=16000)

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            num_workers=2,
            sample_rate=16000,
            augmentation=augmentation,
            processor=self.processor,
            use_synthetic=self.use_synthetic
        )

        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            input_features = batch['input_features'].to(self.device)
            transcripts = batch['transcript']

            labels = self.processor.tokenizer(
                transcripts,
                return_tensors="pt",
                padding=True
            ).input_ids.to(self.device)

            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(
                    input_features=input_features,
                    labels=labels
                )
                loss = outputs.loss / self.gradient_accumulation_steps

            loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            progress_bar.set_postfix({
                'loss': total_loss / num_batches,
                'lr': self.optimizer.param_groups[0]['lr']
            })

        return total_loss / num_batches

    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_features = batch['input_features'].to(self.device)
                transcripts = batch['transcript']

                labels = self.processor.tokenizer(
                    transcripts,
                    return_tensors="pt",
                    padding=True
                ).input_ids.to(self.device)

                labels[labels == self.processor.tokenizer.pad_token_id] = -100

                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    outputs = self.model(
                        input_features=input_features,
                        labels=labels
                    )
                    loss = outputs.loss

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self):
        total_steps = len(self.train_loader) * self.num_epochs // self.gradient_accumulation_steps

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        best_val_loss = float('inf')

        self.logger.info("Starting training...")

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            self.logger.info(
                f"Epoch {epoch}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.logger.info(f"New best validation loss: {val_loss:.4f}")

            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                train_loss,
                val_loss,
                self.output_dir,
                is_best=is_best
            )

        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f}")

        config = {
            'model_name': self.model_name,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'best_val_loss': best_val_loss
        }
        save_config(config, os.path.join(self.output_dir, 'config.json'))

    def save_model(self, save_path: str = None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'final_model')

        self.logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Whisper with LoRA')
    parser.add_argument('--model_name', type=str, default='openai/whisper-small')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./models')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--use_synthetic', action='store_true')
    parser.add_argument('--no_mixed_precision', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    trainer = WhisperLoRATrainer(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_synthetic=args.use_synthetic,
        mixed_precision=not args.no_mixed_precision,
        seed=args.seed
    )

    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    main()
