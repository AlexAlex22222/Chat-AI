import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
device = torch.device('cuda')



all_symbs_enc = {0: 'A', 1: 'a', 2: 'B', 3: 'b', 4: 'C', 5: 'c', 6: 'D', 7: 'd', 8: 'E', 9: 'e', 10: 'F', 11: 'f', 12: 'G', 13: 'g', 14: 'H', 15: 'h', 16: 'I', 17: 'i', 18: 'J', 19: 'j', 20: 'K', 21: 'k', 22: 'L', 23: 'l', 24: 'M', 25: 'm', 26: 'N', 27: 'n', 28: 'O', 29: 'o', 30: 'P', 31: 'p', 32: 'Q', 33: 'q', 34: 'R', 35: 'r', 36: 'S', 37: 's', 38: 'T', 39: 't', 40: 'U', 41: 'u', 42: 'V', 43: 'v', 44: 'W', 45: 'w', 46: 'X', 47: 'x', 48: 'Y', 49: 'y', 50: 'Z', 51: 'z', 52: 'А', 53: 'а', 54: 'Б', 55: 'б', 56: 'В', 57: 'в', 58: 'Г', 59: 'г', 60: 'Д', 61: 'д', 62: 'Е', 63: 'е', 64: 'Ё', 65: 'ё', 66: 'Ж', 67: 'ж', 68: 'З', 69: 'з', 70: 'И', 71: 'и', 72: 'Й', 73: 'й', 74: 'К', 75: 'к', 76: 'Л', 77: 'л', 78: 'М', 79: 'м', 80: 'Н', 81: 'н', 82: 'О', 83: 'о', 84: 'П', 85: 'п', 86: 'Р', 87: 'р', 88: 'С', 89: 'с', 90: 'Т', 91: 'т', 92: 'У', 93: 'у', 94: 'Ф', 95: 'ф', 96: 'Х', 97: 'х', 98: 'Ц', 99: 'ц', 100: 'Ч', 101: 'ч', 102: 'Ш', 103: 'ш', 104: 'Щ', 105: 'щ', 106: 'Ы', 107: 'ы', 108: 'Ь', 109: 'ь', 110: 'Ъ', 111: 'ъ', 112: 'Э', 113: 'э', 114: 'Ю', 115: 'ю', 116: 'Я', 117: 'я', 118: ':', 119: '-', 120: ',', 121: '.', 122: ';', 123: '?', 124: '!', 125: '(', 126: ')', 127: '"', 128: '0', 129: '1', 130: '2', 131: '3', 132: '4', 133: '5', 134: '6', 135: '7', 136: '8', 137: '9', 138: ' ', 139: '&', 140: '#', 141: '=', 142: '+', 143: '–', 144: ':', 145: "'", 146: '*', 147: '/'}

all_symbs = {'A': 0, 'a': 1, 'B': 2, 'b': 3, 'C': 4, 'c': 5, 'D': 6, 'd': 7, 'E': 8, 'e': 9, 'F': 10, 'f': 11, 'G': 12, 'g': 13, 'H': 14, 'h': 15, 'I': 16, 'i': 17, 'J': 18, 'j': 19, 'K': 20, 'k': 21, 'L': 22, 'l': 23, 'M': 24, 'm': 25, 'N': 26, 'n': 27, 'O': 28, 'o': 29, 'P': 30, 'p': 31, 'Q': 32, 'q': 33, 'R': 34, 'r': 35, 'S': 36, 's': 37, 'T': 38, 't': 39, 'U': 40, 'u': 41, 'V': 42, 'v': 43, 'W': 44, 'w': 45, 'X': 46, 'x': 47, 'Y': 48, 'y': 49, 'Z': 50, 'z': 51, 'А': 52, 'а': 53, 'Б': 54, 'б': 55, 'В': 56, 'в': 57, 'Г': 58, 'г': 59, 'Д': 60, 'д': 61, 'Е': 62, 'е': 63, 'Ё': 64, 'ё': 65, 'Ж': 66, 'ж': 67, 'З': 68, 'з': 69, 'И': 70, 'и': 71, 'Й': 72, 'й': 73, 'К': 74, 'к': 75, 'Л': 76, 'л': 77, 'М': 78, 'м': 79, 'Н': 80, 'н': 81, 'О': 82, 'о': 83, 'П': 84, 'п': 85, 'Р': 86, 'р': 87, 'С': 88, 'с': 89, 'Т': 90, 'т': 91, 'У': 92, 'у': 93, 'Ф': 94, 'ф': 95, 'Х': 96, 'х': 97, 'Ц': 98, 'ц': 99, 'Ч': 100, 'ч': 101, 'Ш': 102, 'ш': 103, 'Щ': 104, 'щ': 105, 'Ы': 106, 'ы': 107, 'Ь': 108, 'ь': 109, 'Ъ': 110, 'ъ': 111, 'Э': 112, 'э': 113, 'Ю': 114, 'ю': 115, 'Я': 116, 'я': 117, ':': 144, '-': 119, ',': 120, '.': 121, ';': 122, '?': 123, '!': 124, '(': 125, ')': 126, '"': 127, '0': 128, '1': 129, '2': 130, '3': 131, '4': 132, '5': 133, '6': 134, '7': 135, '8': 136, '9': 137, ' ': 138, '&': 139, '#': 140, '=': 141, '+': 142, '–': 143, "'": 145, '*': 146, '/': 147}

quest = ['Привет', 'Привет.', 'Привет', 'Привет', 'Прив', 'Привет.', 'привет', 'привет', 'Дарова', 'прив', 'прив', 'Привет.', 'Привет, как дела?', 'Привет, как дела?', 'Привет, как дела?', 'Привет, как дела?', 'Привет, как дела?', 'Привет, как дела?', 'Привет, как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Кд?', 'кд', 'Кд', 'кд', 'Что делаешь?', 'Норм, а у тебя?', 'Хорошо, а у тебя?', 'Неплохо, а у тебя?', 'Плохо.', 'Нормально. Как у тебя?', 'Спасибо, неплохо. А как у тебя?', 'Привет, как ты?', 'Как дела?', 'Привет.', 'Дарова.', 'Приветик.', 'Что делаешь?', 'Что расскажешь о себе?', 'Здравствуй.', 'Здравствуй.', 'Здравствуй.', 'Здравствуй.', 'Привет, как ты?', 'Привет, ты как?', 'Привет, как ты?', 'Привет, как ты?', 'Привет, как ты?', 'Привет, как ты?', 'Привет, ты как?', 'Приветик.', 'Приветики.', 'Приветики', 'Приветик.', 'Приветик', 'Приветики', 'Приветики.', 'Здравствуйте.', 'Здравствуйте.', 'Здравствуйте.', 'Привет.', 'Здравствуйте.', 'Здравствуйте.']

answ = ['Привет', 'Привет', 'Прив', 'Привет, как дела?', 'Прив, кд?', 'Привет.', 'Привет', 'Привет.', 'Ну привет.', 'прив', 'Прив', 'Привет, как дела?', 'Привет, хорошо, а у тебя?', 'Хорошо, а у тебя?', 'Привет, хорошо.', 'Привет, неплохо.', 'Неплохо. А у тебя?', 'Нормально, а у тебя?', 'Отлично. А у тебя?', 'Неплохо, а у тебя?', 'Отлично. А у тебя?', 'Прекрасно, а у тебя?', 'Хорошо, а у тебя?', 'Неплохо. А у тебя?', 'Хорошо. А у тебя?', 'Норм, у тебя?', 'норм, у тебя?', 'нормально, у тебя?', 'неплохо, как у тебя?', 'Да вот, с тобой общаюсь.', 'Прекрасно.', 'Хорошо.', 'Спасибо, неплохо.', 'А что так?', 'Спасибо, хорошо.х', 'Тоже хорошо.', 'Привет, неплохо. А ты?', 'Хорошо, а у тебя?', 'И тебе привет.', 'Здравствуй.', 'И тебе привет.', 'Да так, ничего. А ты?', 'Да ничего особенного, я просто ИИ, который создан для чата.', 'Привет.', 'Здравствуй, как дела?', 'Дарова, как дела?', 'Прив.', 'Нормально, спасибо.', 'Привет, хорошо, спасибо.', 'Хорошо, спасибо, а ты?', 'Хорошо, спасибо.', 'Неплохо, спасибо, а ты?', 'Неплохо, а ты?', 'Хорошо, а ты?', 'И тебе привет.', 'Привет-привет.', 'Привет, как дела?', 'Ага, и тебе привет.', 'Привет.', 'И тебе привет.', 'Привет, как дела?', 'Привет.', 'Здравствуйте.', 'Привет, как дела?', 'Ага, и тебе привет.', 'Приветики.', 'Прив.']



def in_tensor(input_list, all_list):
  # lett = ['A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i', 'J', 'j', 'K',
  #           'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's', 'T', 't', 'U', 'u',
  #           'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z', 'А', 'а', 'Б', 'б', 'В', 'в', 'Г', 'г', 'Д', 'д', 'Е',
  #           'е', 'Ё', 'ё', 'Ж', 'ж', 'З', 'з', 'И', 'и', 'Й', 'й', 'К', 'к', 'Л', 'л', 'М', 'м', 'Н', 'н', 'О', 'о',
  #           'П', 'п', 'Р', 'р', 'С', 'с', 'Т', 'т', 'У', 'у', 'Ф', 'ф', 'Х', 'х', 'Ц', 'ц', 'Ч', 'ч', 'Ш', 'ш', 'Щ',
  #           'щ', 'Ы', 'ы', 'Ь', 'ь', 'Ъ', 'ъ', 'Э', 'э', 'Ю', 'ю', 'Я', 'я']
  # nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  # symbs = [':', '-', ',', '.', ';', '?', '!', '(', ')', '"', "'"]
  max_l = 40
  ts_list = []
  ins_list = []
  for w in input_list:
    ins_list2 = []
    for i in w:
        g = all_list[i]
        ins_list2.append(g)
        # if i in lett:
        #     ins_list2.append(1.)
        # else:
        #     ins_list2.append(0.)
        # if i in nums:
        #       ins_list2.append(1.)
        # else:
        #     ins_list2.append(0.)
        # if i in symbs:
        #         ins_list2.append(1.)
        # else:
        #     ins_list2.append(0.)
        #ins_list2.append(0.)
    if w == input_list[-1]:
      pass
    else:
      ins_list2.append(139)
    ins_list.append(ins_list2)
    # for g in ins_list:
    # #for g in x:
    #   count = len(g)
    #   if count > max_l:
    #       max_l = count
    #   else:
    #       pass
    for g in ins_list:
    #for g in x:
      count = len(g)
      if count < max_l:
        while len(g) < max_l:
            g.append(0)
  
  ts_list.append(ins_list)
  for ins_list in ts_list:
    # for b in ins_list:
      for i in range(20 - len(ins_list)):
        jk = []
        for h in range(max_l):
          jk.append(round(0.))
        ins_list.append(jk)
  
  return ts_list 



def encode(outpit, all_list):
  g = ""
  outpit = outpit.cpu().detach().numpy()
  for i in outpit:
   for h in i:
    hb = ""
    for b in h:
     output = round(abs(b), 3)
     if output == 0.:
      continue
     enc = all_list[output]
     hb = hb+enc
    g = g+hb
  return g




def rand_choice(answ, quest):
  qlen = len(quest)
  hk = random.randint(0, qlen - 1)
  input = quest[hk]
  output = answ[hk]
  return input, output




def backwinp(input_list, all_symbs):
  outp_list = []
  for i in input_list:
    for g in i:
      outp_list.append(all_symbs[g])
  while len(outp_list)<2000:
      outp_list.append(round(0.))
  return outp_list     
  




class RecurrentNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RecurrentNet, self).__init__()
        self.lstm = nn.LSTM(input_size = 20, hidden_size = 2000, num_layers = 30, batch_first=True)
        self.Softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(2000, 25, dtype = torch.int32)
#        nn.Softmax()
#        self.fc2 = nn.Linear(20, 4980, dtype = torch.float64)
#        nn.Softmax()
#        self.fc3 = nn.Linear(4980, 20, dtype = torch.float64)
#        nn.Softmax()
#        self.fc4 = nn.Linear(4100, 3790, dtype = torch.float64)
#        nn.Softmax()
#        self.fc5 = nn.Linear(3790, 3096, dtype = torch.float64)
#        nn.Softmax()
#        self.fc6 = nn.Linear(3096, 2800, dtype = torch.float64)
#        nn.Softmax()
#        self.fc7 = nn.Linear(2800, 2500, dtype = torch.float64)
#        nn.Softmax()
#        self.fc8 = nn.Linear(2100, 1980)
#        nn.Softmax()
#        self.fc9 = nn.Linear(1980, 1024)
#        nn.Softmax()
#        self.fc10 = nn.Linear(1024, 998)
#        nn.Softmax()
#        self.fc11 = nn.Linear(998, 950)
#        nn.Softmax()
#        self.fc12 = nn.Linear(950, 1640)
#        nn.Softmax()
#        self.fc13 = nn.Linear(1640, 2500)
#        nn.Softmax()
#        self.fc14 = nn.Linear(2500, 3000)
#        nn.Softmax()
#        self.fc15 = nn.Linear(3000, 3500)
#        nn.Softmax()
#        self.fc16 = nn.Linear(3500, 4000)
#        nn.Softmax()
#        self.fc17 = nn.Linear(4000, 4500)
#        nn.Softmax()
#        self.fc18 = nn.Linear(4500, 5000)
#        nn.Softmax()
#        self.fc19 = nn.Linear(5000, 5500)
#        nn.Softmax()
#        self.fc20 = nn.AdaptiveAvgPool1d(20)
    def forward(self, x):
#        out = x
        out, (ht1, ct1)  = self.lstm(x, 2000)
        out = self.dropout(out)
        out = self.fc(out)
#        out, _ = self.rnn2(out)
#        out = self.fc2(out)
#        out = self.fc3(out)
#        out = self.fc4(out)
#        out = self.fc5(out)
#        #print(out)
#        out = self.fc6(out)
#        #print(out)
#        out = self.fc7(out)
#        out = self.fc8(out)
#        out = self.fc9(out)
#        out = self.fc10(out)
#        out = self.fc11(out)
#        out = self.fc12(out)
#        out = self.fc13(out)
#        out = self.fc14(out)
#        out = self.fc15(out)
#        out = self.fc16(out)
#        out = self.fc17(out)
#        out = self.fc18(out)
#        out = self.fc19(out)
#        out = self.fc20(out)
        return out

    def init_hidden(self, batch_size=1):
        return (torch.zeros(30, batch_size, 2000, requires_grad=True).to(device),
               torch.zeros(30, batch_size, 2000, requires_grad=True).to(device))


def train(net, criterion, optimizer, device, scheduler1, quest, answ, all_symbs):
     loss_avg = []
     epochs = 50000
     for epoch in range(epochs):
         net.train()
         inputs, targets = rand_choice(answ, quest)
         inputs = in_tensor(inputs, all_symbs)
         targets = in_tensor(targets, all_symbs)
         inputs = torch.tensor(inputs, dtype = torch.int32)
         targets = torch.tensor(targets, dtype = torch.int32)
         inputs, targets = inputs.to(device), targets.to(device)
         hidden = net.init_hidden(16)
         output = net(quest)
         loss = criterion(output, targets)
         loss.backward()
         optimizer.step()
         optimizer.zero_grad(l
         loss_avg.append(loss.item())
         if len(loss_avg) >= 50:
          mean_loss = np.mean(loss_avg)
          print(f'Loss: {mean_loss}')
          scheduler1.step(mean_loss)
          loss_avg = []
          net.eval()
          predicted_text = output
          print(predicted_text)



#a = encode(x, all_symbs_enc)
#print(a)     
input_text = input("E: ").split(" ")
x = in_tensor(input_text, all_symbs)
#print(x)
#outpt_text = input("EO: ").split(" ")
#xo = in_tensor(outpt_text, all_symbs)
gx = torch.tensor(x, dtype = torch.int32)
#print(gx.type())
#size = len(x[0][0])
# size2 = len(x)
# print(size)
print(gx)
gx = gx.view(20, 40, 1)
print(gx)
net = RecurrentNet(40, 4032, 40)
#net.load_state_dict(torch.load('model_weights.pth'))
#net.eval()
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
# inputs = torch.randn(2, 5, 5)
#kl = net.forward(gx)
#print(kl)
#print(kl.size())
#x = in_tensor(input_text, all_symbs)
#targets = torch.tensor(x, dtype = torch.float64)
#print(targets.type())
#print(targets)
#print(len(targets))
net.cuda()
#inputs = gx.cuda() 
#targets = targets.cuda()




try:
  train(net, criterion, optimizer, device, scheduler1, quest, answ, all_symbs)
  torch.save(net.state_dict(), 'model_weights.pth')
except KeyboardInterrupt:
  torch.save(net.state_dict(), 'model_weights.pth')
  KeyboardInterrupt()
#print(encode(kl, all_symbs_enc))
