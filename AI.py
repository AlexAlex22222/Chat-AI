import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
device = torch.device('cpu')

all_symbs = {'A': 0.001, 'a': 0.002, 'B': 0.003, 'b': 0.004, 'C': 0.005, 'c': 0.006, 'D': 0.007, 'd': 0.008, 'E': 0.009, 'e': 0.01, 'F': 0.011, 'f': 0.012, 'G': 0.013, 'g': 0.014, 'H': 0.015, 'h': 0.016, 'I': 0.017, 'i': 0.018, 'J': 0.019, 'j': 0.02, 'K': 0.021, 'k': 0.022, 'L': 0.023, 'l': 0.024, 'M': 0.025, 'm': 0.026, 'N': 0.027, 'n': 0.028, 'O': 0.029, 'o': 0.03, 'P': 0.031, 'p': 0.032, 'Q': 0.033, 'q': 0.034, 'R': 0.035, 'r': 0.036, 'S': 0.037, 's': 0.038, 'T': 0.039, 't': 0.04, 'U': 0.041, 'u': 0.042, 'V': 0.043, 'v': 0.044, 'W': 0.045, 'w': 0.046, 'X': 0.047, 'x': 0.048, 'Y': 0.049, 'y': 0.05, 'Z': 0.051, 'z': 0.052, 'А': 0.053, 'а': 0.054, 'Б': 0.055, 'б': 0.056, 'В': 0.057, 'в': 0.058, 'Г': 0.059, 'г': 0.06, 'Д': 0.061, 'д': 0.062, 'Е': 0.063, 'е': 0.064, 'Ё': 0.065, 'ё': 0.066, 'Ж': 0.067, 'ж': 0.068, 'З': 0.069, 'з': 0.07, 'И': 0.071, 'и': 0.072, 'Й': 0.073, 'й': 0.074, 'К': 0.075, 'к': 0.076, 'Л': 0.077, 'л': 0.078, 'М': 0.079, 'м': 0.08, 'Н': 0.081, 'н': 0.082, 'О': 0.083, 'о': 0.084, 'П': 0.085, 'п': 0.086, 'Р': 0.087, 'р': 0.088, 'С': 0.089, 'с': 0.09, 'Т': 0.091, 'т': 0.092, 'У': 0.093, 'у': 0.094, 'Ф': 0.095, 'ф': 0.096, 'Х': 0.097, 'х': 0.098, 'Ц': 0.099, 'ц': 0.1, 'Ч': 0.101, 'ч': 0.102, 'Ш': 0.103, 'ш': 0.104, 'Щ': 0.105, 'щ': 0.106, 'Ы': 0.107, 'ы': 0.108, 'Ь': 0.109, 'ь': 0.11, 'Ъ': 0.111, 'ъ': 0.112, 'Э': 0.113, 'э': 0.114, 'Ю': 0.115, 'ю': 0.116, 'Я': 0.117, 'я': 0.118, ':': 0.119, '-': 0.12, ',': 0.121, '.': 0.122, ';': 0.123, '?': 0.124, '!': 0.125, '(': 0.126, ')': 0.127, '"': 0.128, '0': 0.129, '1': 0.13, '2': 0.131, '3': 0.132, '4': 0.133, '5': 0.134, '6': 0.135, '7': 0.136, '8': 0.137, '9': 0.138, ' ': 0.139, "'": 0.14}

all_symbs_enc = {0.001: 'A', 0.002: 'a', 0.003: 'B', 0.004: 'b', 0.005: 'C', 0.006: 'c', 0.007: 'D', 0.008: 'd', 0.009: 'E', 0.01: 'e', 0.011: 'F', 0.012: 'f', 0.013: 'G', 0.014: 'g', 0.015: 'H', 0.016: 'h', 0.017: 'I', 0.018: 'i', 0.019: 'J', 0.02: 'j', 0.021: 'K', 0.022: 'k', 0.023: 'L', 0.024: 'l', 0.025: 'M', 0.026: 'm', 0.027: 'N', 0.028: 'n', 0.029: 'O', 0.03: 'o', 0.031: 'P', 0.032: 'p', 0.033: 'Q', 0.034: 'q', 0.035: 'R', 0.036: 'r', 0.037: 'S', 0.038: 's', 0.039: 'T', 0.04: 't', 0.041: 'U', 0.042: 'u', 0.043: 'V', 0.044: 'v', 0.045: 'W', 0.046: 'w', 0.047: 'X', 0.048: 'x', 0.049: 'Y', 0.05: 'y', 0.051: 'Z', 0.052: 'z', 0.053: 'А', 0.054: 'а', 0.055: 'Б', 0.056: 'б', 0.057: 'В', 0.058: 'в', 0.059: 'Г', 0.06: 'г', 0.061: 'Д', 0.062: 'д', 0.063: 'Е', 0.064: 'е', 0.065: 'Ё', 0.066: 'ё', 0.067: 'Ж', 0.068: 'ж', 0.069: 'З', 0.07: 'з', 0.071: 'И', 0.072: 'и', 0.073: 'Й', 0.074: 'й', 0.075: 'К', 0.076: 'к', 0.077: 'Л', 0.078: 'л', 0.079: 'М', 0.08: 'м', 0.081: 'Н', 0.082: 'н', 0.083: 'О', 0.084: 'о', 0.085: 'П', 0.086: 'п', 0.087: 'Р', 0.088: 'р', 0.089: 'С', 0.09: 'с', 0.091: 'Т', 0.092: 'т', 0.093: 'У', 0.094: 'у', 0.095: 'Ф', 0.096: 'ф', 0.097: 'Х', 0.098: 'х', 0.099: 'Ц', 0.1: 'ц', 0.101: 'Ч', 0.102: 'ч', 0.103: 'Ш', 0.104: 'ш', 0.105: 'Щ', 0.106: 'щ', 0.107: 'Ы', 0.108: 'ы', 0.109: 'Ь', 0.11: 'ь', 0.111: 'Ъ', 0.112: 'ъ', 0.113: 'Э', 0.114: 'э', 0.115: 'Ю', 0.116: 'ю', 0.117: 'Я', 0.118: 'я', 0.119: ':', 0.12: '-', 0.121: ',', 0.122: '.', 0.123: ';', 0.124: '?', 0.125: '!', 0.126: '(', 0.127: ')', 0.128: '"', 0.129: '0', 0.13: '1', 0.131: '2', 0.132: '3', 0.133: '4', 0.134: '5', 0.135: '6', 0.136: '7', 0.137: '8', 0.138: '9', 0.139: ' ', 0.14: "'"}



def in_tensor(input_list, all_list):
  lett = ['A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i', 'J', 'j', 'K',
            'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's', 'T', 't', 'U', 'u',
            'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z', 'А', 'а', 'Б', 'б', 'В', 'в', 'Г', 'г', 'Д', 'д', 'Е',
            'е', 'Ё', 'ё', 'Ж', 'ж', 'З', 'з', 'И', 'и', 'Й', 'й', 'К', 'к', 'Л', 'л', 'М', 'м', 'Н', 'н', 'О', 'о',
            'П', 'п', 'Р', 'р', 'С', 'с', 'Т', 'т', 'У', 'у', 'Ф', 'ф', 'Х', 'х', 'Ц', 'ц', 'Ч', 'ч', 'Ш', 'ш', 'Щ',
            'щ', 'Ы', 'ы', 'Ь', 'ь', 'Ъ', 'ъ', 'Э', 'э', 'Ю', 'ю', 'Я', 'я']

  nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

  symbs = [':', '-', ',', '.', ';', '?', '!', '(', ')', '"', "'"]
  max_l = 0
  ts_list = []
  for w in input_list:
    ins_list = []
    for i in w:
        ins_list2 = []
        g = all_list[i]
        ins_list2.append(g)
        if i in lett:
            ins_list2.append(1.)
        else:
            ins_list2.append(0.)
        if i in nums:
            ins_list2.append(1.)
        else:
            ins_list2.append(0.)
        if i in symbs:
            ins_list2.append(1.)
        else:
            ins_list2.append(0.)
        if i == ' ':
            ins_list2.append(1.)
        else:
            ins_list2.append(0.)
        ins_list.append(ins_list2)
    if w == input_list[-1]:
      pass
    else:
      ins_list.append([all_list[" "], 0., 0., 0., 0.])
    ts_list.append(ins_list)
  
  for g in ts_list:
    #for g in x:
      count = len(g)
      if count > max_l:
          max_l = count
      else:
          continue
  for g in ts_list:
    #for g in x:
      count = len(g)
      if count < max_l:
        while len(g) < max_l:
            g.append([0., 0., 0., 0., 0.])
  return ts_list 


def encode(input_list, all_list):
  g = ""
  for i in input_list:
    h = ""
    for k in i:
      for y in k:
       if y == 0:
        continue
       enc = all_list[k]
      h = h+enc
    g = g+h
  return g

#a = encode(x, all_symbs_enc)
#print(a)     

input_text = input("E: ").split(" ")
x = in_tensor(input_text, all_symbs)
print(x)
#outpt_text = input("EO: ").split(" ")
#xo, seq_len2 = in_tensor(outpt_text, all_symbs)
gx = torch.tensor(x)
size = len(x[0][0])
size2 = len(x)
print(size)
print(gx)

class RecurrentNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, size2):
        super(RecurrentNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        nn.Softmax()
        self.fc = nn.Linear(512, 240)
        nn.Softmax()
        self.fc2 = nn.Linear(3, 20)



    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.fc2(out[:, 1])
        return out


# def train(net, criterion, optimizer, inputs, targets, device):
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     epochs = 100
#     for epoch in range(epochs):
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#     acc = 100. * correct / total
#     train_loss /= len(train_loader.dataset)

#     return train_loss, acc

net = RecurrentNet(size, 512, 140, 20, size2)
#criterion = nn.MSELoss()
#optimizer = optim.Adam(net.parameters(), lr=0.01)
#inputs = torch.randn(2, 5, 5)
kl = net.forward(gx)
print(kl)
print(kl.size())
#targets = ogx
#train_loss, acc = train(net, criterion, optimizer, inputs, targets, device)
