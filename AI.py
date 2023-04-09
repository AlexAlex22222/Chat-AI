import os
#assert os.environ['COLAB_TPU_ADDR'] # 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'
#!pip install cloud-tpu-client==0.10 torch==2.0.0 torchvision==0.15.1 https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp39-cp39-linux_x86_64.whl
#!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
#!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
#!sudo apt install python3.7 -y
#!pip freeze
#!update-alternatives --config python3
#!pip install torch
#!pip install ray
#import ray
!export XLA_USE_BF16=1
!export TRIM_GRAPH_CHECK_FREQUENCY=100000 
!export TRIM_GRAPH_SIZE=1000000
!export XLA_USE_32BIT_LONG=1
!export XLA_IR_DEBUG=1
!export XLA_HLO_DEBUG=1
!export TF_CPP_VMODULE=tensor=5
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import pickle
import random, math




device ='cuda' if torch.cuda.is_available() == True else 'cpu'

c_dev = 'cpu'

torch.manual_seed(123)

weights = torch.randn(156, 20).to(device)

all_symbs = {'A': 1, 'a': 2, 'B': 3, 'b': 4, 'C': 5, 'c': 6, 'D': 7, 'd': 8, 'E': 9, 'e': 10, 'F': 11, 'f': 12, 'G': 13, 'g': 14, 'H': 15, 'h': 16, 'I': 17, 'i': 18, 'J': 19, 'j': 20, 'K': 21, 'k': 22, 'L': 23, 'l': 24, 'M': 25, 'm': 26, 'N': 27, 'n': 28, 'O': 29, 'o': 30, 'P': 31, 'p': 32, 'Q': 33, 'q': 34, 'R': 35, 'r': 36, 'S': 37, 's': 38, 'T': 39, 't': 40, 'U': 41, 'u': 42, 'V': 43, 'v': 44, 'W': 45, 'w': 46, 'X': 47, 'x': 48, 'Y': 49, 'y': 50, 'Z': 51, 'z': 52, 'А': 53, 'а': 54, 'Б': 55, 'б': 56, 'В': 57, 'в': 58, 'Г': 59, 'г': 60, 'Д': 61, 'д': 62, 'Е': 63, 'е': 64, 'Ё': 65, 'ё': 66, 'Ж': 67, 'ж': 68, 'З': 69, 'з': 70, 'И': 71, 'и': 72, 'Й': 73, 'й': 74, 'К': 75, 'к': 76, 'Л': 77, 'л': 78, 'М': 79, 'м': 80, 'Н': 81, 'н': 82, 'О': 83, 'о': 84, 'П': 85, 'п': 86, 'Р': 87, 'р': 88, 'С': 89, 'с': 90, 'Т': 91, 'т': 92, 'У': 93, 'у': 94, 'Ф': 95, 'ф': 96, 'Х': 97, 'х': 98, 'Ц': 99, 'ц': 100, 'Ч': 101, 'ч': 102, 'Ш': 103, 'ш': 104, 'Щ': 105, 'щ': 106, 'Ы': 107, 'ы': 108, 'Ь': 109, 'ь': 110, 'Ъ': 111, 'ъ': 112, 'Э': 113, 'э': 114, 'Ю': 115, 'ю': 116, 'Я': 117, 'я': 118, ':': 119, '-': 120, ',': 121, '.': 122, ';': 149, '?': 124, '!': 125, '(': 126, ')': 127, '"': 128, '0': 129, '1': 130, '2': 131, '3': 132, '4': 133, '5': 134, '6': 135, '7': 136, '8': 137, '9': 138, ' ': 139, '#': 140, '$': 141, '%': 142, '+': 143, '×': 144, '÷': 145, '=': 146, '/': 147, '_': 148, "'": 150, '^': 151, '&': 152, '>': 153, '<': 154, '~': 155, '`': 156, '*': 157}

all_symbs_enc = {1: 'A', 2: 'a', 3: 'B', 4: 'b', 5: 'C', 6: 'c', 7: 'D', 8: 'd', 9: 'E', 10: 'e', 11: 'F', 12: 'f', 13: 'G', 14: 'g', 15: 'H', 16: 'h', 17: 'I', 18: 'i', 19: 'J', 20: 'j', 21: 'K', 22: 'k', 23: 'L', 24: 'l', 25: 'M', 26: 'm', 27: 'N', 28: 'n', 29: 'O', 30: 'o', 31: 'P', 32: 'p', 33: 'Q', 34: 'q', 35: 'R', 36: 'r', 37: 'S', 38: 's', 39: 'T', 40: 't', 41: 'U', 42: 'u', 43: 'V', 44: 'v', 45: 'W', 46: 'w', 47: 'X', 48: 'x', 49: 'Y', 50: 'y', 51: 'Z', 52: 'z', 53: 'А', 54: 'а', 55: 'Б', 56: 'б', 57: 'В', 58: 'в', 59: 'Г', 60: 'г', 61: 'Д', 62: 'д', 63: 'Е', 64: 'е', 65: 'Ё', 66: 'ё', 67: 'Ж', 68: 'ж', 69: 'З', 70: 'з', 71: 'И', 72: 'и', 73: 'Й', 74: 'й', 75: 'К', 76: 'к', 77: 'Л', 78: 'л', 79: 'М', 80: 'м', 81: 'Н', 82: 'н', 83: 'О', 84: 'о', 85: 'П', 86: 'п', 87: 'Р', 88: 'р', 89: 'С', 90: 'с', 91: 'Т', 92: 'т', 93: 'У', 94: 'у', 95: 'Ф', 96: 'ф', 97: 'Х', 98: 'х', 99: 'Ц', 100: 'ц', 101: 'Ч', 102: 'ч', 103: 'Ш', 104: 'ш', 105: 'Щ', 106: 'щ', 107: 'Ы', 108: 'ы', 109: 'Ь', 110: 'ь', 111: 'Ъ', 112: 'ъ', 113: 'Э', 114: 'э', 115: 'Ю', 116: 'ю', 117: 'Я', 118: 'я', 119: ':', 120: '-', 121: ',', 122: '.', 123: ';', 124: '?', 125: '!', 126: '(', 127: ')', 128: '"', 129: '0', 130: '1', 131: '2', 132: '3', 133: '4', 134: '5', 135: '6', 136: '7', 137: '8', 138: '9', 139: ' ', 140: '#', 141: '$', 142: '%', 143: '+', 144: '×', 145: '÷', 146: '=', 147: '/', 148: '_', 149: ';', 150: "'", 151: '^', 152: '&', 153: '>', 154: '<', 155: '~', 156: '`', 157: '*'}

# quest = ['Привет', 'Привет', 'Привет', 'Привет', 'Привет', 'Привет.', 'Привет.', 'Привет.', 'Привет', 'Привет', 'Привет', 'Пртвет.', 'Привет, как ты?', 'Привет, как ты?', 'Приветики.', 'Прив', 'Привет.', 'Прив', 'Привет.', 'Прив', 'Привет.', 'привет', 'привет', 'Дарова', 'прив', 'прив', 'Привет.', 'Привет, как дела?', 'Привет, как дела?', 'Привет, как дела?', 'Здравствуй, как дела?', 'Привет, как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Привет, как дела?', 'Привет, как дела?', 'Привет, как дела?', 'Привет.', 'Прив', 'Привет, как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Привет, как дела?', 'Привет, как дела?', 'Привет, как дела?', 'Привет, как дела?', 'Привет, как дела?', 'Привет, как дела?', 'Привет, как дела?', 'Привет, как дела?', 'Привет, как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Как дела?', 'Кд?', 'кд', 'Кд', 'кд', 'Что делаешь?', 'Норм, а у тебя?', 'Хорошо, а у тебя?', 'Неплохо, а у тебя?', 'Плохо.', 'Нормально. Как у тебя?', 'Спасибо, неплохо. А как у тебя?', 'Привет, как ты?', 'Как дела?', 'Привет.', 'Дарова.', 'Приветик.', 'Что делаешь?', 'Что расскажешь о себе?', 'Здравствуй.', 'Здравствуй.', 'Здравствуй.', 'Здравствуй.', 'Привет, как ты?', 'Привет, ты как?', 'Привет, как ты?', 'Привет, как ты?', 'Привет, как ты?', 'Привет, как ты?', 'Привет, ты как?', 'Приветик.', 'Приветики.', 'Приветики', 'Приветики, как дела?', 'Приветики, как дела?', 'Приветики, как дела?', 'Приветик.', 'Приветик', 'Приветики', 'Приветики.', 'Здравствуйте.', 'Здравствуйте.', 'Здравствуйте.', 'Привет.', 'Здравствуйте.', 'Здравствуйте.', 'Хорошо, а у тебя?', 'Хорошо.', 'Прекрасно, а у тебя?', 'Замечательно.', 'Нормально, а у тебя?', 'Хорошо, а у тебя?', 'Хорошо, а у тебя?', 'Хорошо, а у тебя?', 'Хорошо, а у тебя?', 'Хорошо, а у тебя?', 'Хорошо, а у тебя?', 'Хорошо.', 'Хорошо.', 'Нормально, а у тебя?', 'Нормально, а у тебя?', 'Плохо.', 'Плохо.', 'Не очень, а у тебя?', 'Плохо.', 'Плохо.', 'Плохо.', 'Плохо', 'Плохо', 'Хорошо, спасибо.', 'Хорошо.', 'Хорошо, спасибо.', 'Нормально вроде.', 'Хорошо, а у тебя?', 'Хорошо, спасибо, а как у тебя дела?', 'Хорошо, спасибо, а у тебя как?', 'Хорошо, а у тебя как?', 'Хорошо, а у тебя?', 'Хорошо, а у тебя как?']

# answ = ['Привет', 'Здравствуйте.', 'Здравствуй', 'Здравствуйте', 'Здравствуй', 'Здравствуйте.', 'Здравствуй.', 'Привет', 'Прив', 'Привет, как дела?', 'Привет, как у тебя дела?', 'Привет, как дела?', 'Спасибо, всё хорошо, а ты как?', 'Привет, да всё прекрасно, а ты как?', 'И тебе привет, как дела?', 'Привет, как у тебя дела?', 'Приветики.', 'Приветики.', 'Привет, как у тебя дела?', 'Прив, кд?', 'Привет.', 'Привет', 'Привет.', 'Ну привет.', 'прив', 'Прив', 'Привет, как дела?', 'Привет, хорошо, а у тебя?', 'Привет, неплохо, а у тебя?', 'Привет, нормально, а у тебя?', 'Здравствуй, неплохо, а у тебя?', 'Здравствуйте, неплохо, а у вас?', 'Привет, хорошо, а у тебя?', 'Привет, неплохо, а у тебя?', 'Привет, нормально, а у тебя?', 'Хорошо, а у тебя?', 'Нормально, а у тебя?', 'Прекрасно, а у тебя?', 'Хорошо, как у тебя?', 'Да хорошо, а у тебя?', 'Хорошо, а у тебя как?', 'Хорошо, а как у тебя?', 'И тебе привет.', 'И тебе привет.', 'Да просто замечательно, а у тебя?', 'Всё хорошо, а как твои дела?', 'Спасибо, хорошо, а у тебя как?', 'Спасибо, хорошо, а как у тебя?', 'Привет, у меня хорошо, а как твои дела?', 'Спасибо, у меня всё хорошо, а как у тебя дела?', 'Спасибо, у меня неплохо, а как у тебя?', 'Привет, спасибо, хорошо. А у тебя как?', 'Привет, неплохо, а твои как?', 'Да неплохо, спасибо. А у тебя как?', 'Хорошо, а у тебя?', 'Привет, хорошо.', 'Привет, неплохо.', 'Неплохо. А у тебя?', 'Нормально, а у тебя?', 'Отлично. А у тебя?', 'Неплохо, а у тебя?', 'Отлично. А у тебя?', 'Прекрасно, а у тебя?', 'Хорошо, а у тебя?', 'Неплохо. А у тебя?', 'Хорошо. А у тебя?', 'Спасибо, ничего, а у тебя?', 'Да хорошо, спасибо, а у тебя как?', 'Спасибо, всё хорошо, а у тебя как?', 'Спасибо, всё прекрасно, а у тебя как дела?', 'Да всё хорошо, а у тебя как?', 'Норм, у тебя?', 'норм, у тебя?', 'нормально, у тебя?', 'неплохо, как у тебя?', 'Да вот, с тобой общаюсь.', 'Прекрасно.', 'Хорошо.', 'Спасибо, неплохо.', 'А что так?', 'Спасибо, хорошо.', 'Тоже хорошо.', 'Привет, неплохо. А ты?', 'Хорошо, а у тебя?', 'И тебе привет.', 'Здравствуй.', 'И тебе привет.', 'Да так, ничего. А ты?', 'Да ничего особенного, я просто ИИ, который создан для чата.', 'Привет.', 'Здравствуй, как дела?', 'Дарова, как дела?', 'Прив.', 'Нормально, спасибо.', 'Привет, хорошо, спасибо.', 'Хорошо, спасибо, а ты?', 'Хорошо, спасибо.', 'Неплохо, спасибо, а ты?', 'Неплохо, а ты?', 'Хорошо, а ты?', 'И тебе привет.', 'Привет-привет.', 'Привет, как дела?', 'Привет, хорошо, а у тебя?', 'Приветик, неплохо.', 'Здравствуй, хорошо.', 'Ага, и тебе привет.', 'Привет.', 'И тебе привет.', 'Привет, как дела?', 'Привет.', 'Здравствуйте.', 'Привет, как дела?', 'Ага, и тебе привет.', 'Приветики.', 'Прив.', 'Тоже хорошо, спасибо.', 'Прекрасно.', 'Тоже замечательно, спасибо.', 'Это хорошо.', 'Спасибо, хорошо.', 'Хорошо, спасибо.', 'Неплохо, спасибо.', 'Спасибо, хорошо.', 'Прекрасно, спасибо.', 'Спасибо, неплохо.', 'Неплохо.', 'Замечательно.', 'Это хорошо.', 'Спасибо, хорошо.', 'Хорошо, спасибо.', 'А что так?', 'Почему?', 'Почему не очень?', 'А почему плохо?', 'Да? А что так?', 'А что случилось?', 'Почему так?', 'Вот как, а почему?', 'Вот и прекрасно.', 'Вот и прекрасно.', 'Вот и замечательно.', 'Вот и хорошо.', 'Рад слышать. У меня всё хорошо, спасибо.', 'Отлично всё, спасибо.', 'Спасибо, у меня всё тоже хорошо.', 'Спасибо, тоже хорошо.', 'Спасибо, неплохо.', 'Всё хорошо, спасибо.']




def encode_in_tns1(inp_list):
  word_l = 20
  sent_l = 90
  sec_l_len = 20
  final_list = []
  sec_list = []
  if len(inp_list) == 0:
    pass
  else:
   for item in inp_list:
    wordchr_list = []
    
    for charc in item:
       wordchr_list.append(all_symbs[charc])
       
    if item != inp_list[-1]:
      wordchr_list.append(all_symbs[' '])
      
    if len(wordchr_list)<word_l:
      while len(wordchr_list)<word_l:
        wordchr_list.append(0)
        
    final_list.append(wordchr_list)
    
  if len(final_list)<sent_l:
    while len(final_list) < sent_l:
      empt_list = []
      while len(empt_list)<word_l:
        empt_list.append(0)
      final_list.append(empt_list)
      
  return final_list




class RecurrentNet(nn.Module):
    def __init__(self, input_size):
        super(RecurrentNet, self).__init__()
        self.input_size = input_size
        self.softmax = nn.Softmax()
        self.fc = nn.LSTM(input_size, 400, 3)
        self.fc1 = nn.LSTM(100, 100, 3)
        self.fc2 = nn.Linear(100, input_size)
        self.hid_t = nn.Linear(100, 100)
        self.c_t = nn.Linear(400, 100)                
        self.fc3 = nn.Linear(400, 100)


    def forward(self, x, hid):
        x, (h1, c1) = self.fc(x)
        ht1, ct1 = hid
        c2 = self.c_t(c1)
        x = self.fc3(x)
        x, (ht2, ct2) = self.fc1(x, (ht1,  c2))
        x = self.fc2(x)
        return x, (ht2, ct2)


class Embedng(nn.Module):
  def __init__(self):
    super(Embedng, self).__init__()
    self.vocab = list(all_symbs)
    self.enc_vocab = list(all_symbs_enc)
    self.embed = nn.Embedding.from_pretrained(weights).to(device)
    self.cos = nn.CosineSimilarity(dim=2)
  def to_embed_seq(self, seqs):
    seqs = torch.LongTensor(seqs).to(device)
    emb_seq = self.embed(seqs)
    return emb_seq
  def unembed(self, embedded_sequence):
        weights = self.embed.state_dict()['weight']
        weights = weights.transpose(0,1).unsqueeze(0).unsqueeze(0)
        e_sequence = embedded_sequence.unsqueeze(3).data
        cosines = self.cos(e_sequence, weights)
        _, indexes = torch.topk(cosines, 1, dim=2)
        words = []
        for word in indexes:
            word_l = ''
            for char_index in word:
              if char_index == 0:
                continue
              else:
                word_l += all_symbs_enc[int(char_index)]
            if word_l != '':
              words.append(word_l)
            else:
              continue
        return words



embedding = Embedng()

embedding = embedding.to(device)


def train(net, device):
     net.train()
     
     scheph = 50
     scheph1 = scheph
     losseph = 10
     losseph1 = losseph
     sprange = len(answ)
     range_lst = []
     loss_lst = []
     crit = nn.MSELoss()
     for i in range(0, sprange-1):
       range_lst.append(i)
     loss_avg = []
     epochs = 10000
     optimizer = torch.optim.SGD(net.parameters(), lr = 1.5)
     scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    patience=5, 
    verbose=False, 
    factor=0.5
)
     for epoch in range(epochs):
      hid = (torch.zeros(3, 20, 100).to(device), torch.zeros(3, 20, 100).to(device))
      random.shuffle(range_lst)
      loss_in_data = []
      print(epoch)
      for i in range_lst:
         inputs, targets = quest[i].split(" "), answ[i].split(" ")
         inputs = encode_in_tns1(inputs)
         inputs = embedding.to_embed_seq(inputs)
         targets = encode_in_tns1(targets)
         targets = embedding.to_embed_seq(targets)
         inputs, targets = inputs.to(device), targets.to(device)
         output, (hf1, cf1) = net(inputs, hid)
         loss = crit(output, targets)
         loss.backward()
         optimizer.step()
         hf1 = hf1.detach()
         cf1 = cf1.detach()
         hid = (hf1, cf1)
         optimizer.zero_grad()
         loss_avg.append(loss.item())
         loss_in_data.append(loss.item())
      print(embedding.unembed(output))
      loss_lst.append(sum(loss_in_data)/len(loss_in_data))
      if epoch == losseph:
        print("Loss: ", sum(loss_in_data)/len(loss_in_data))
        losseph += losseph1
     plt.plot(loss_lst)
     plt.show()
     print(embedding.unembed(output))
      
net = RecurrentNet(20)

if torch.cuda.is_available() == True:
  print("GPU")
  net.cuda()
net = net.to(device)

envi = "g"

if envi == "k":
 try:
  if os.path.exists("/kaggle/input/weight/model_weights.pth"):
   print("Loading weights...")
   net.load_state_dict(torch.load('/kaggle/input/weight/model_weights.pth', map_location=torch.device(device))) 
   net.eval()
 except RuntimeError:
  print("Wrong weigts")
  pass
else:
 try:
  if os.path.exists("/content/model_weights.pth"):
   print("Loading weights...")
   net.load_state_dict(torch.load('/content/model_weights.pth', map_location=torch.device(device))) 
   net.eval()
 except RuntimeError:
  print("Wrong weigts")
  pass


inp = input("Вы: ").split(" ")
inp = encode_in_tns1(inp)
inp = embedding.to_embed_seq(inp)
hid = spec_hid(inp)
output, cont = net.forward(inp, hid)
answer = embedding.unembed(output)
strng = ""
for i in answer:
 strng = strng + i
print(strng)
train(net, device)


# try:
#   train(net, device)
#   torch.save(net.state_dict(), 'model_weights.pth')
# except KeyboardInterrupt:
#   torch.save(net.state_dict(), 'model_weights.pth')
#   KeyboardInterrupt()


###############################################################################################################################
