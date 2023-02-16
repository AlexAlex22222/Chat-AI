import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

all_symbs = ['A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i', 'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z', 'А', 'а', 'Б', 'б', 'В', 'в', 'Г', 'г', 'Д', 'д', 'Е', 'е', 'Ё', 'ё', 'Ж', 'ж', 'З', 'з', 'И', 'и', 'Й', 'й', 'К', 'к', 'Л', 'л', 'М', 'м', 'Н', 'н', 'О', 'о', 'П', 'п', 'Р', 'р', 'С', 'с', 'Т', 'т', 'У', 'у', 'Ф', 'ф', 'Х', 'х', 'Ц', 'ц', 'Ч', 'ч', 'Ш', 'ш', 'Щ', 'щ', 'Ы', 'ы', 'Ь', 'ь', 'Ъ', 'ъ', 'Э', 'э', 'Ю', 'ю', 'Я', 'я', ':', '-', ',', '.', ';', '?', '!', '(', ')', '"', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', " "]

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(1, 4)
        self.layer2 = nn.Linear(4, 512)
        self.layer3 = nn.Linear(512, 240)
        self.layer4 = nn.Linear(240, 128)
        self.layer5 = nn.Linear(128, 10)
        self.layer6 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = self.layer6(x)
        return x

def train_neural_network(x, y, num_epochs):
    model = NeuralNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    train_loss_history = []
    for epoch in range(num_epochs):
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_history.append(loss.item())
    return model, train_loss_history

def plot_training_results(train_loss_history):
    plt.plot(train_loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()



def numeraise(input_list, all_list):
    numeraited_dict = {elem:num for num, elem in enumerate(all_list)}
    numeraited_list = [numeraited_dict[i] for item in input_list for i in item]
    return numeraited_list

def unnumeraise(input_list, all_list):
    unnumeraited_dict = {num:elem for num, elem in enumerate(all_list)}
    unnumeraited_list = [unnumeraited_dict[item] for item in input_list]
    return unnumeraited_list
  

input_text = input('Enter input: ')
output_text = input('Enter output: ')

input_saved = numeraise(input_text, all_symbs)
output_saved = numeraise(output_text, all_symbs)
#input_saved = input_saved.astype(np.int64)
#output_saved = output_saved.astype(np.int64)
print(input_saved)
print(output_saved)

X = torch.tensor(input_saved).reshape(1, -1)
y = torch.tensor(output_saved).reshape(1, -1)

G = NeuralNet()
G.forward(X)
#unnum = unnumeraise(input_saved, all_symbs)
#print(unnum)

#train_loss_history = train_neural_network(X, y, num_epochs=1000)
#plot_training_results(train_loss_history)