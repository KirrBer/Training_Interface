import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import random
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import copy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Normalize_Model():
    def __init__(self,
                 batch_size=16,
                 epochs=8,
                 lr=2e-4,
                 weight_decay=0.01,
                 patience=3,
                 min_delta=0.05,
                 task="normalize skill",
                 start_model="./start_model",
                 scheduler=False):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = T5ForConditionalGeneration.from_pretrained(start_model, tie_word_embeddings=True)
        self.tokenizer = T5Tokenizer.from_pretrained(start_model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.train_results_loss = []
        self.train_results_accuracy = []
        self.train_results_validate_loss = []
        self.accuracy = None
        self.patience = patience
        self.min_delta = min_delta
        self.task = task
        self.start_model = start_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.best_model_state = None
        self.scheduler = scheduler
        self.best_epoch = None
    def answer(self, x, **kwargs):
        inputs = self.tokenizer(x, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            hypotheses = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(hypotheses[0], skip_special_tokens=True)

    def train(self, pairs: list[tuple[str, str]], test_pairs=None, aug=False):
        if aug:
            pairs = self.augmentation(pairs)
        if self.task:
            pairs = [[self.task+": "+p[0], p[1]] for p in pairs]
            if test_pairs:
                test_pairs = [[self.task+": "+p[0], p[1]] for p in test_pairs]
        logging.info(f"Starting training with {len(pairs)} pairs for {self.epochs} epochs")
        cur_patience = self.patience
        for epoch in range(self.epochs):

            if self.scheduler:
                x = (epoch+1) / self.epochs
                cur_lr = round((1/(1+np.exp(6*(x**2-0.4)))+0.1)*self.lr, 6)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = cur_lr

            random.shuffle(pairs)
            self.model.train()
            losses = []

            for i in range(0, int(len(pairs) / self.batch_size)):
                batch = pairs[i * self.batch_size: (i + 1) * self.batch_size]
                x = self.tokenizer([p[0] for p in batch], return_tensors='pt', padding=True).to(self.model.device)
                y = self.tokenizer([p[1] for p in batch], return_tensors='pt', padding=True).to(self.model.device)
                y.input_ids[y.input_ids == 0] = -100
                loss = self.model(
                    input_ids=x.input_ids,
                    attention_mask=x.attention_mask,
                    labels=y.input_ids,
                    decoder_attention_mask=y.attention_mask,
                    return_dict=True
                ).loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(loss.item())

            self.train_results_loss.append(np.mean(losses))
            if not test_pairs and epoch > 0 and self.train_results_loss[-1] - self.train_results_loss[-2] < self.min_delta:
                    cur_patience -= 1
            logging.info(f"Epoch {epoch+1}/{self.epochs}: Loss = {self.train_results_loss[-1]:.4f}")
            if test_pairs:
                self.model.eval()
                self.train_results_accuracy.append(sum([test_pairs[i][1]==self.answer(test_pairs[i][0]) for i in range(len(test_pairs))])/len(test_pairs))
                self.accuracy = self.train_results_accuracy[-1]
                val_losses = []

                for i in range(0, len(test_pairs), self.batch_size):
                    batch = test_pairs[i:i+self.batch_size]
                    x = self.tokenizer([p[0] for p in batch], return_tensors='pt', padding=True).to(self.model.device)
                    y = self.tokenizer([p[1] for p in batch], return_tensors='pt', padding=True).to(self.model.device)
                    y.input_ids[y.input_ids == 0] = -100

                    with torch.no_grad():
                        loss = self.model(
                            input_ids=x.input_ids,
                            attention_mask=x.attention_mask,
                            labels=y.input_ids,
                            decoder_attention_mask=y.attention_mask,
                        ).loss
                    val_losses.append(loss.item())

                self.train_results_validate_loss.append(np.mean(val_losses))

                if self.train_results_validate_loss[-1] < min(self.train_results_validate_loss):
                    self.best_model_state = copy.deepcopy(self.model.state_dict())

                if epoch>0 and self.train_results_validate_loss[-1] - self.train_results_validate_loss[-2] < self.min_delta:
                    cur_patience -= 1
                    if cur_patience == 0:
                        logging.info("Early stopping triggered")
                        break

                logging.info(f"Epoch {epoch+1}/{self.epochs}: Accuracy = {self.accuracy:.4f}")
                logging.info(f"Epoch {epoch+1}/{self.epochs}: Validate Loss = {self.train_results_validate_loss[-1]:.4f}")
            if cur_patience == 0:
                logging.info("Early stopping triggered")
                break
        self.model.eval()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        logging.info("Training completed")

    def save(self, output="new_model"):
        logging.info(f"Saving model to {output}")
        self.model.save_pretrained(output+'/last_model')
        self.tokenizer.save_pretrained(output+'/last_model')
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            self.model.save_pretrained(output+'/best_model')
            self.tokenizer.save_pretrained(output+'/best_model')

    def test(self, pairs: list[tuple[str, str]], folds=5, aug=False):
        logging.info(f"Starting cross-validation with {folds} folds on {len(pairs)} pairs")
        random.shuffle(pairs)
        self.train_results_loss = [0]*self.epochs
        self.train_results_accuracy = [0]*self.epochs
        self.train_results_validate_loss = [0]*self.epochs

        for i in range(folds):
            test_data = pairs[i*(len(pairs)//folds):(i+1)*(len(pairs)//folds)]
            train_data = pairs[0:(i*len(pairs)//folds)]+pairs[(i+1)*(len(pairs)//folds):]
            test_model = Normalize_Model(
                patience=-1,
                start_model=self.start_model,
                batch_size=self.batch_size,
                epochs=self.epochs,
                lr=self.lr,
                weight_decay=self.weight_decay,
                min_delta=self.min_delta,
                task=self.task,
                scheduler=self.scheduler
            )

            test_model.train(train_data, test_pairs=test_data, aug=aug)
            self.train_results_loss = [x+y for x,y in zip(self.train_results_loss,test_model.train_results_loss)]
            self.train_results_accuracy = [x+y for x,y in zip(self.train_results_accuracy,test_model.train_results_accuracy)]
            self.train_results_validate_loss = [x+y for x,y in zip(self.train_results_validate_loss,test_model.train_results_validate_loss)]
            logging.info(f"Fold {i+1}/{folds}: Accumulated Accuracy = {[round(x, 4) for x in self.train_results_accuracy]}, Accumulated validate Loss = {[round(x, 4) for x in self.train_results_validate_loss]}")
            del test_model

        logging.info(f"Cross-validation completed. Average Accuracy per epoch: {[round(x/folds, 4) for x in self.train_results_accuracy]}, Average validate Loss per epoch: {[round(x/folds, 4) for x in self.train_results_validate_loss]}")
        self.train_results_loss = [x/folds for x in self.train_results_loss]
        self.train_results_accuracy = [x/folds for x in self.train_results_accuracy]
        self.train_results_validate_loss = [x/folds for x in self.train_results_validate_loss]
        self.accuracy = self.train_results_accuracy[-1]
        self.best_epoch = self.train_results_accuracy.index(max(self.train_results_accuracy))+1
        self.graph()
        return self.accuracy

    def graph(self):
        if self.train_results_accuracy:
            fig, axs = plt.subplots(3,1)
            axs[0].plot(range(1,self.epochs+1), self.train_results_loss)
            axs[0].set_ylabel("loss")
            axs[0].set_xlabel("Эпоха")
            axs[0].grid()

            axs[1].plot(range(1,self.epochs+1), self.train_results_accuracy)
            axs[1].set_ylabel("accuracy")
            axs[1].set_xlabel("epoch")
            axs[1].grid()

            axs[2].plot(range(1,self.epochs+1), self.train_results_validate_loss)
            axs[2].set_ylabel("validate loss")
            axs[2].set_xlabel("epoch")
            axs[2].grid()
            plt.tight_layout()
            plt.show()
        elif self.train_results_loss:
            plt.plot(range(1,self.epochs+1), self.train_results_loss)
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.grid()
            plt.tight_layout()
            plt.show()
        else:
            print("model hasn't been trained")

    def augmentation(self, training_data: list[tuple[str, str]]):
        rand_elems = random.sample(training_data, int(len(training_data)/6))
        rand_chars = ['.',',',' ','1','2','3','4','5','6','7','8','9','.','.','.',',',',','-','-']
        for i in rand_elems:
            char = random.choice(rand_chars)
            res = [i[0]+char,i[1]]
            training_data.append(res)

        for i in range(int(len(training_data)/6)):
            rand_elem = random.randint(0, len(training_data)-1-i)
            rand_char_num = random.randint(0, len(training_data[rand_elem][0]))
            new_data_elem = [training_data[rand_elem][0][:rand_char_num]+training_data[rand_elem][0][rand_char_num+1:],training_data[rand_elem][1]]
            training_data.append(new_data_elem)
        for i in range(int(len(training_data)/6)):
            rand_elem = random.randint(0, len(training_data)-1-i)
            rand_char_num = random.randint(0, len(training_data[rand_elem][0]))
            rand_char = random.randint(ord('a'), ord('z'))
            new_data_elem = [training_data[rand_elem][0][:rand_char_num]+chr(rand_char)+training_data[rand_elem][0][rand_char_num:],training_data[rand_elem][1]]
            training_data.append(new_data_elem)
        for i in range(int(len(training_data)/6)):
            rand_elem = random.randint(0, len(training_data)-1-i)
            new_data_elem = [training_data[rand_elem][0].upper(),training_data[rand_elem][1]]
            training_data.append(new_data_elem)
        for i in range(int(len(training_data)/6)):
            rand_elem = random.randint(0, len(training_data)-1-i)
            new_data_elem = [training_data[rand_elem][0].lower(),training_data[rand_elem][1]]
            training_data.append(new_data_elem)

        return training_data


def load_data(file_path):
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',', 1)  # Split only on first comma
                if len(parts) == 2:
                    pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the Normalize Model")
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Mode: train or test')
    parser.add_argument('--data_file', required=True, help='Path to TXT file with training data')
    parser.add_argument('--test_file', help='Path to TXT file with validate data')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=8, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=0.01, help='Min delta for early stopping')
    parser.add_argument('--task', type=str, default='normalize skill', help='Task prefix')
    parser.add_argument('--start_model', type=str, default='./start_model', help='Path to start model')
    parser.add_argument('--output', type=str, default='new_model', help='Output path for saving model')
    parser.add_argument('--aug', action='store_true', help='Use augmentation')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--scheduler', action='store_true', help='Use learning rate scheduler')

    
    args = parser.parse_args()
    
    pairs = load_data(args.data_file)
    model = Normalize_Model(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        task=args.task,
        start_model=args.start_model,
        scheduler=args.scheduler
    )
    
    if args.mode == 'train':
        model.train(pairs, aug=args.aug, test_pairs=load_data(args.test_file) if args.test_file else None)
        model.save(args.output)
    elif args.mode == 'test':
        accuracy = model.test(pairs, folds=args.folds, aug=args.aug)
        print(f"Final accuracy: {accuracy}")
        print(f"Best epoch: {model.best_epoch}")
        if args.output:
            model.train(pairs, aug=args.aug, epochs=model.best_epoch)
            model.save(args.output)






