import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import random
import numpy as np
import matplotlib.pyplot as plt


class Normalize_Model():
    def __init__(self, 
                 batch_size=16, 
                 epochs=16, 
                 lr=2e-4, 
                 weight_decay=0.01, 
                 patience=3, 
                 min_delta=0.01, 
                 task="normalize skill", 
                 start_model="./start_model"):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = T5ForConditionalGeneration.from_pretrained(start_model)
        self.tokenizer = T5Tokenizer.from_pretrained(start_model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.train_results_loss = []
        self.train_results_accuracy = []
        self.accuracy = None
        self.patience = patience
        self.min_delta = min_delta
        self.task = task
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
        for epoch in range(self.epochs):
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
            if epoch>0 and self.train_results_loss[-1] - self.train_results_loss[-2] < self.min_delta:
                self.patience -= 1
            if test_pairs:
                self.model.eval()
                self.train_results_accuracy.append(sum([test_pairs[i][1]==self.answer(test_pairs[i][0]) for i in range(len(test_pairs))])/len(test_pairs))
                self.accuracy = self.train_results_accuracy[-1]
            if self.patience == 0:
                print("Early stopping triggered")
                break
        self.patience = 3
        self.model.eval()
    def save(self, output="new_model"):
        self.model.save_pretrained(output)
        self.tokenizer.save_pretrained(output)
    def test(self, pairs: list[tuple[str, str]], branches=5, aug=False):
        random.shuffle(pairs)
        self.train_results_loss = [0]*self.epochs
        self.train_results_accuracy = [0]*self.epochs
        for i in range(branches):
            test_data = pairs[i*(len(pairs)//branches):(i+1)*(len(pairs)//branches)]
            train_data = pairs[0:(i*len(pairs)//branches)]+pairs[(i+1)*(len(pairs)//branches):]
            test_model = Normalize_Model(patience=-1,
                                         start_model=self.start_model,
                                         batch_size=self.batch_size, 
                                         epochs=self.epochs, 
                                         lr=self.lr, 
                                         weight_decay=self.weight_decay, 
                                         min_delta=self.min_delta, 
                                         task=self.task
                                        )
            test_model.train(train_data, test_pairs=test_data, aug=aug)
            self.train_results_loss = [x+y for x,y in zip(self.train_results_loss,test_model.train_results_loss)]
            self.train_results_accuracy = [x+y for x,y in zip(self.train_results_accuracy,test_model.train_results_accuracy)]
            print(self.train_results_accuracy)
            print(self.train_results_loss)
            del test_model
        self.train_results_loss = [x/branches for x in self.train_results_loss]
        self.train_results_accuracy = [x/branches for x in self.train_results_accuracy]
        self.accuracy = self.train_results_accuracy[-1]
        self.graph()
        return self.accuracy
    def graph(self):
        if self.train_results_accuracy:
            fig, axs = plt.subplots(2,1)
            axs[0].plot(range(1,self.epochs+1), self.train_results_loss)
            axs[0].set_ylabel("loss")
            axs[0].set_xlabel("Эпоха")
            axs[0].grid()

            axs[1].plot(range(1,self.epochs+1), self.train_results_accuracy)
            axs[1].set_ylabel("accuracy")
            axs[1].set_xlabel("epoch")
            axs[1].grid() 
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
        rand_elems = random.sample(training_data, int(len(training_data)/5))
        rand_chars = ['.',',',' ','1','2','3','4','5','6','7','8','9','.','.','.',',',',','-','-']
        for i in rand_elems:
            char = random.choice(rand_chars)
            res = [i[0]+char,i[1]]
            training_data.append(res)
        for i in range(int(len(training_data)/5)):
            rand_elem = random.randint(0, len(training_data)-1-i)
            rand_char_num = random.randint(0, len(training_data[rand_elem][0]))
            new_data_elem = [training_data[rand_elem][0][:rand_char_num]+training_data[rand_elem][0][rand_char_num+1:],training_data[rand_elem][1]]
            training_data.append(new_data_elem)
        for i in range(int(len(training_data)/5)):
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








