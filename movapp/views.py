from django.shortcuts import render
from .models import Text
from .forms import TextForm
import requests
import re
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

ZERO = 0
UNDEF = 1
LEN=200

vocab = {}
f = open('./imdb.vocab')
for i in range(10000):
	d = f.readline()[:-1]
	vocab[d] = i + 2

class SimpleModel(nn.Module): # код модели для обучения
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(10002, 200)
        self.norm1 = nn.BatchNorm1d(200)
        self.lstm = nn.LSTM(200, 512, 2, dropout=0.2, batch_first=True)
        self.norm2 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(512, 256)
        self.norm3 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.norm4 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.long()
        x = self.norm1(self.embedding(x))
        x =  self.lstm(x)[0][:, -1, :]
        x = self.dropout1(self.norm2(x))
        x = F.relu(self.norm3(self.fc1(x)))
        x = F.relu(self.norm4(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = SimpleModel()
model.load_state_dict(torch.load('./trained_model.pth', map_location='cpu'))
model.eval()

def preprocess_data(data):
	data = re.sub(r'([\.\,\!\?\-])', r' \1 ', data) # добавляем пробелы
	data = re.sub(r'([^\x00-\x7f])', r'', data) # удяляем не алфавитные символы
	data = re.sub(r'([\,\.\!\?\-\"\#\$\%\&\(\)\*\+\/\:\;\<\=\>\@\[\\\]\^\_\`\{\|\}\~1234567890])', '', data) # удаляем знаки препинания и цифры
	data = data.split(' ') # разбиваем текст на слова
	vec = []
	counter = 0
	for word in data: # для каждого слова
		if counter == LEN: # лимит на кол-во элементов в векторе
			break
		if len(word) < 2: 
			continue
		if (word[0] == "'"): # дополнительно убираем одинарные кавычки по краям (ранее они не удалились, чтобы оставить их в артиклях)
			word = word[1:]
		if len(word) < 2:
			continue
		if (word[-1] == "'"):
			word = word[:-1]
		if len(word) < 2:
			continue
		word = word.lower() # нижний регистр
		if word in vocab:
			vec.append(vocab[word]) # добавляем в вектор код слова
		else:
			vec.append(UNDEF) # добавляем в вектор код UNDEF
		counter += 1
	dop = LEN - len(vec)
	for i in range(dop): # заполняем вектор нулями до длины 200 (если длина меньше)
		vec.append(ZERO)
	return torch.tensor(vec)

def index(request):
	form = TextForm()
	res = ''
	if (request.method == 'POST'):
		form = TextForm(request.POST)
		res = preprocess_data(request.POST.dict()['name'])
		out = model(res.unsqueeze(0))
		out = out.squeeze().item()
		if out >= 0.5:
			res = 'Positive'
		else:
			res = 'Negative'

	context = {'form' : form, 'result' : res}

	return render(request, 'movapp/index.html', context)