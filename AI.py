#!jupyter nbconvert --to script script.ipynb
#!pip install colorama
#GH12-2309hjkl:5647jkhfg
import os
import contextlib, io
from torch import Tensor, TensorType, randn, manual_seed, cuda, optim, save, IntTensor
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import math
from colorama import Fore, Back, Style, init

init(convert=True)

device ='cuda' if cuda.is_available() == True else 'cpu'
c_dev = 'cpu'
manual_seed(1111)
weights = randn(157, 7).to(device)

all_symbs = {'A': 1, 'a': 2, 'B': 3, 'b': 4, 'C': 5, 'c': 6, 'D': 7, 'd': 8, 'E': 9, 'e': 10, 'F': 11, 'f': 12, 'G': 13, 'g': 14, 'H': 15, 'h': 16, 'I': 17, 'i': 18, 'J': 19, 'j': 20, 'K': 21, 'k': 22, 'L': 23, 'l': 24, 'M': 25, 'm': 26, 'N': 27, 'n': 28, 'O': 29, 'o': 30, 'P': 31, 'p': 32, 'Q': 33, 'q': 34, 'R': 35, 'r': 36, 'S': 37, 's': 38, 'T': 39, 't': 40, 'U': 41, 'u': 42, 'V': 43, 'v': 44, 'W': 45, 'w': 46, 'X': 47, 'x': 48, 'Y': 49, 'y': 50, 'Z': 51, 'z': 52, 'А': 53, 'а': 54, 'Б': 55, 'б': 56, 'В': 57, 'в': 58, 'Г': 59, 'г': 60, 'Д': 61, 'д': 62, 'Е': 63, 'е': 64, 'Ё': 65, 'ё': 66, 'Ж': 67, 'ж': 68, 'З': 69, 'з': 70, 'И': 71, 'и': 72, 'Й': 73, 'й': 74, 'К': 75, 'к': 76, 'Л': 77, 'л': 78, 'М': 79, 'м': 80, 'Н': 81, 'н': 82, 'О': 83, 'о': 84, 'П': 85, 'п': 86, 'Р': 87, 'р': 88, 'С': 89, 'с': 90, 'Т': 91, 'т': 92, 'У': 93, 'у': 94, 'Ф': 95, 'ф': 96, 'Х': 97, 'х': 98, 'Ц': 99, 'ц': 100, 'Ч': 101, 'ч': 102, 'Ш': 103, 'ш': 104, 'Щ': 105, 'щ': 106, 'Ы': 107, 'ы': 108, 'Ь': 109, 'ь': 110, 'Ъ': 111, 'ъ': 112, 'Э': 113, 'э': 114, 'Ю': 115, 'ю': 116, 'Я': 117, 'я': 118, ':': 119, '-': 120, ',': 121, '.': 122, ';': 149, '?': 124, '!': 125, '(': 126, ')': 127, '"': 128, '0': 129, '1': 130, '2': 131, '3': 132, '4': 133, '5': 134, '6': 135, '7': 136, '8': 137, '9': 138, ' ': 139, '#': 140, '$': 141, '%': 142, '+': 143, '×': 144, '÷': 145, '=': 146, '/': 147, '_': 148, "'": 150, '^': 151, '&': 152, '>': 153, '<': 154, '~': 155, '`': 156, '*': 157}

all_symbs_enc = {1: 'A', 2: 'a', 3: 'B', 4: 'b', 5: 'C', 6: 'c', 7: 'D', 8: 'd', 9: 'E', 10: 'e', 11: 'F', 12: 'f', 13: 'G', 14: 'g', 15: 'H', 16: 'h', 17: 'I', 18: 'i', 19: 'J', 20: 'j', 21: 'K', 22: 'k', 23: 'L', 24: 'l', 25: 'M', 26: 'm', 27: 'N', 28: 'n', 29: 'O', 30: 'o', 31: 'P', 32: 'p', 33: 'Q', 34: 'q', 35: 'R', 36: 'r', 37: 'S', 38: 's', 39: 'T', 40: 't', 41: 'U', 42: 'u', 43: 'V', 44: 'v', 45: 'W', 46: 'w', 47: 'X', 48: 'x', 49: 'Y', 50: 'y', 51: 'Z', 52: 'z', 53: 'А', 54: 'а', 55: 'Б', 56: 'б', 57: 'В', 58: 'в', 59: 'Г', 60: 'г', 61: 'Д', 62: 'д', 63: 'Е', 64: 'е', 65: 'Ё', 66: 'ё', 67: 'Ж', 68: 'ж', 69: 'З', 70: 'з', 71: 'И', 72: 'и', 73: 'Й', 74: 'й', 75: 'К', 76: 'к', 77: 'Л', 78: 'л', 79: 'М', 80: 'м', 81: 'Н', 82: 'н', 83: 'О', 84: 'о', 85: 'П', 86: 'п', 87: 'Р', 88: 'р', 89: 'С', 90: 'с', 91: 'Т', 92: 'т', 93: 'У', 94: 'у', 95: 'Ф', 96: 'ф', 97: 'Х', 98: 'х', 99: 'Ц', 100: 'ц', 101: 'Ч', 102: 'ч', 103: 'Ш', 104: 'ш', 105: 'Щ', 106: 'щ', 107: 'Ы', 108: 'ы', 109: 'Ь', 110: 'ь', 111: 'Ъ', 112: 'ъ', 113: 'Э', 114: 'э', 115: 'Ю', 116: 'ю', 117: 'Я', 118: 'я', 119: ':', 120: '-', 121: ',', 122: '.', 123: ';', 124: '?', 125: '!', 126: '(', 127: ')', 128: '"', 129: '0', 130: '1', 131: '2', 132: '3', 133: '4', 134: '5', 135: '6', 136: '7', 137: '8', 138: '9', 139: ' ', 140: '#', 141: '$', 142: '%', 143: '+', 144: '×', 145: '÷', 146: '=', 147: '/', 148: '_', 149: ';', 150: "'", 151: '^', 152: '&', 153: '>', 154: '<', 155: '~', 156: '`', 157: '*'}

quest1 = ['Привет', 'Привет.', 'привет', 'прив', 'Здравствуй', 'Здравствуй.', 'Здравствуйте.', 'Здравствуйте', 'Здарова.', 'Здарова', 'Привет, как дела?', 'Как дела?', 'Кд?', 'кд', 'Тоже хорошо.', 'Хорошо.', 'Неплохо.', 'Не очень.', 'Плохо.', 'Что делаешь?', 'чд', 'Чд?', 'Пока.', 'До свидания.']

answ1 = ['Привет.', 'Привет.', 'Привет.', 'Привет.', 'Здравствуй.', 'Здравствуй.', 'Привет.', 'Привет.', 'Привет.', 'Привет.', 'Привет, хорошо, а у тебя как?', 'Хорошо, а у тебя как?', 'Хорошо, а у тебя как?', 'Хорошо, а у тебя как?', 'Ну вот и замечательно.', 'Я рад за тебя.', 'Это хорошо.', 'А почему?', 'Почему же?', 'Да так, ничего, а ты?', 'Ничего, а ты?', 'Да ничего, а ты?', 'До свидания.', 'Пока.']

quest = ['Привет', 'Привет.', 'привет', 'прив', 'Здравствуй', 'Здравствуй.', 'Здравствуйте.', 'Здравствуйте', 'Здорово.', 'Здорово', 'Привет, как дела?', 'Как дела?', 'Кд?', 'кд', 'Тоже хорошо.', 'Хорошо.', 'Неплохо.', 'Не очень.', 'Плохо.', 'Что делаешь?', 'чд', 'Чд?', 'Пока.', 'До свидания.', 'Сколько будет 2+2?', 'Сколько будет 2×1?', 'Сколько будет 2×2?', 'Сколько будет 2×3?', 'Сколько будет 2×4?', 'Сколько будет 2×5?', 'Сколько будет 2×6?', 'Сколько будет 2×7?', 'Сколько будет 2×8?', 'Сколько будет 2×9?', 'Сколько будет 2×10?', 'Сколько будет 1+1?', 'Сколько будет 1+2?', 'Сколько будет 1+3?', 'Сколько будет 1+4?', 'Сколько будет 1+5?', 'Сколько будет 1+6?', 'Сколько будет 1+7?', 'Сколько будет 1+8?', 'Сколько будет 1+9?', 'Сколько будет 3×8?', 'Сколько будет 4×9?', 'Сколько будет 9×9?', 'Сколько будет 6×7?', 'Как называется человек, который пишет программы?', 'Как стать богатым?', 'Что можешь рассказать о себе?', 'Ты как?', 'Как ты?', 'Ты умеешь считать?', 'Ты живой?', 'Ты разумный?', 'Всё хорошо?', 'Назови все буквы русского алфавита.', 'Назови все цифры в твоём словаре.', 'Назови все цифры в твоём словаре словами.', 'Ты нейросеть?', 'Назови все буквы английского алфавита.', 'Как называется самая известная раскладка клавиатуры?']

answ = ['Привет.', 'Привет.', 'Привет.', 'Привет.', 'Здравствуй.', 'Здравствуй.', 'Привет.', 'Привет.', 'Привет.', 'Привет.', 'Привет, хорошо, а у тебя как?', 'Хорошо, а у тебя как?', 'Хорошо, а у тебя как?', 'Хорошо, а у тебя как?', 'Ну вот и замечательно.', 'Я рад за тебя.', 'Это хорошо.', 'А почему?', 'Почему же?', 'Да так, ничего, а ты?', 'Ничего, а ты?', 'Да ничего, а ты?', 'До свидания.', 'Пока.', 'Ответ: 4.', 'Ответ: 2.', 'Ответ: 4.', 'Ответ: 6.', 'Ответ: 8.', 'Ответ: 10.', 'Ответ: 12.', 'Ответ: 14.', 'Ответ: 16.', 'Ответ: 18.', 'Ответ: 20.', 'Ответ: 2.', 'Ответ: 3.', 'Ответ: 4.', 'Ответ: 5.', 'Ответ: 6.', 'Ответ: 7.', 'Ответ: 8.', 'Ответ: 9.', 'Ответ: 10.', 'Ответ: 24.', 'Ответ: 36.', 'Ответ: 81.', 'Ответ: 42.', 'Программист.', 'Либо выйграть в лотерею, либо открыть бизнес, либо работать.', 'Я языковая модель, ИИ, созданный для чата.', 'Всё хорошо.', 'Всё нормально.', 'Думаю да.', 'Нет, хотя не знаю.', 'Не знаю, возможно да.', 'Да.', 'а, б, в, г, д, е, ё, ж, з, и, й, к, л, м, н, о, п, р, с, т, у, ф, х, ц, ч, ш, щ, ь, ы, ъ, э, ю, я.', '1, 2, 3, 4, 5, 6, 7, 8, 9, 0.', 'Один, два, три, четыре, пять, шесть, семь, восемь, девять, ноль.', 'Да.', 'a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z.', 'QWERTY.']

def system_message_dec(func_to_dec):
  def decor(*args, **kwargs):
    x = io.StringIO()
    try:
      with contextlib.redirect_stdout(x):
        ret = func_to_dec(*args, **kwargs)
      if ret != None:
          print(Fore.RED + f"System error: < \"{ret}\" >", Style.RESET_ALL)
      else:
          print(Fore.YELLOW + f"System message: < \"{x.getvalue().strip()}\" >", Style.RESET_ALL)       
    except Exception as err:
      print(Fore.RED + f"System error: < \"{err}\" >", Style.RESET_ALL)
  return decor



def output_enc(inp_list):
    all_len = 300
    fin_list = []
    #for item in inp_list:
    for charct in inp_list:
            fin_list.append(all_symbs[charct])
    while len(fin_list) < all_len:
        fin_list.append(0)
    return fin_list


def encode_in_tns1(inp_list):
  word_l = 20
  sent_l = 80
  sec_l_len = 20
  final_list = []
  sec_list = []
  if len(inp_list) == 0:
    pass
  else:
   for item in inp_list:
    i = 1
    wordchr_list = []
    
    for charc in item:
       wordchr_list.append(round(((all_symbs[charc]-i)/20/158)*80, 3))
       i +=1
       
    if item != inp_list[-1]:
      wordchr_list.append(round(((all_symbs[' ']-i)/20/158)*80, 3))
      i+=1
      
    if len(wordchr_list)<word_l:
      while len(wordchr_list)<word_l:
        wordchr_list.append(round(((0-i)*20/158)*80, 3))
        i+=1
              
    final_list.append(list(wordchr_list))
    
  if len(final_list)<sent_l:
    while len(final_list) < sent_l:
      i = 1
      empt_list = []
      while len(empt_list)<word_l:
        empt_list.append(round(((0-i)*20/158)*80, 3))
        i+=1
      final_list.append(empt_list)
      
  return final_list
  


class RecurrentNet(nn.Module):
    def __init__(self, input_size):
        super(RecurrentNet, self).__init__()
        self.input_size = input_size
        #self.transf = nn.Transformer(d_model = 105, nhead = 7, num_encoder_layers = 5)
        #self.mem = torch.randn(20, 20, 11)
        #layer = nn.TransformerEncoderLayer(d_model = input_size, nhead = 11)
        #self.enc = nn.TransformerEncoder(layer, 2)
        #layer2 = nn.TransformerDecoderLayer(d_model = input_size, nhead = 11)
        #self.dec = nn.TransformerDecoder(layer2, 2)
        #self.embed = nn.Embedding(len(all_symbs), input_size, padding_idx = 10)
        #self.cos = nn.CosineSimilarity(dim=2)
        #self.fc = nn.LSTM(input_size, 100, 10)
        self.fc = nn.Linear(input_size, 300)
        self.fc1 = nn.RNN(300, 300, 5)
#         self.fc2 = nn.Linear(100, 200)
#         self.fc3 = nn.Linear(200, 500)
#         self.fc4 = nn.Linear(500, 1000)
#         self.fc5 = nn.Linear(1000, 2000)
#         self.fc6 = nn.Linear(2000, 1000)
#         self.fc7= nn.Linear(1000, 700)
#         self.fc8 = nn.Linear(700, 500)
#         self.fc9 = nn.Linear(500, 100)
        #self.fc1 = nn.Linear(400, 100)
        #self.fc2 = nn.LSTM(100, 100, 3)
        #self.h_t = nn.Linear(1000, 20)
        #self.c_t = nn.Linear(1000, 20)                
        self.fc10 = nn.Linear(300, input_size)
        #self.rev_h = nn.Linear(20, 1000)
        #self.rev_c = nn.Linear(20, 1000)
    
    
    def forward(self, x):
        #e = self.enc(x)
        x = self.fc(x)
        #x = self.softmax(x)
        x, _ = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         x = self.fc4(x)
#         x = self.fc5(x)
#         x = self.fc6(x)
#         x = self.fc7(x)
#         x = self.fc8(x)
#         x = self.fc9(x)
        #h2 = self.h_t(h1)
        #c2 = self.c_t(c1)
        #x = self.fc3(x)
        #x, (ht2, ct2) = self.fc1(x, (h2,  c2))
        #x = self.fc1(x)
        #x, hidn = self.fc2(x, hid)
        x = self.fc10(x)
        #x = self.dec(x, e)
        #x = self.softmax(x)
        return x
    
    
    
    
class Embedng(nn.Module):
  def __init__(self):
    super(Embedng, self).__init__()
    #self.vocab = list(all_symbs)
    self.embed = nn.Embedding.from_pretrained(weights).to(device)
    self.cos = nn.CosineSimilarity(dim=2)
  def to_embed_seq(self, seqs):
    seqs = torch.IntTensor(seqs).to(device)
    emb_seq = self.embed(seqs)
    return emb_seq
  def unembed(self, embedded_sequence):
        weights = self.embed.state_dict()['weight']
        weights = weights.transpose(0,1).unsqueeze(0).unsqueeze(0)
        e_sequence = embedded_sequence.unsqueeze(2).data
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



net = RecurrentNet(7)

@system_message_dec
def check_cuda():
  if cuda.is_available() == True:
    print("CUDA available.")
  else:
    print("CUDA isn't available.")

net = net.to(device)
#net = nn.DataParallel(net)
envi = "other"

@system_message_dec
def load_weights(envi):
  #net.load_state_dict(torch.load('/content/model_weights.pth', map_location=torch.device(device))) 
  if envi == "ke":
    try:
      if os.path.exists("/kaggle/input/weight/model_weights.pth"):
        print("Loading weights...")
        net.load_state_dict(torch.load('/kaggle/input/weight/model_weights.pth', map_location=torch.device(device))) 
        net.eval()
      else: return "Weights don't exist in external Kaggle directory."
    except RuntimeError:
      return "Wrong weigts"
      pass
  elif envi =="ki":
    if os.path.exists("/kaggle/working/model_weights.pth"):
      print("Loading weights...")
      net.load_state_dict(torch.load('/kaggle/working/model_weights.pth', map_location=torch.device(device))) 
      net.eval()
    else: return "Weights don't exist in internal Kaggle directory."
  else:
    try:
      if os.path.exists("model_weights.pth"):
        print("Loading weights...")
        net.load_state_dict(torch.load('model_weights.pth', map_location=torch.device(device))) 
        net.eval()
      elif os.path.exists("/model_weights.pth"):
        print("Loading weights...")
        net.load_state_dict(torch.load('/model_weights.pth', map_location=torch.device(device))) 
        net.eval()
      else:
        return "Weights don't exist in this directory."
    except RuntimeError:
        return "Wrong weigts"
        pass
  


def answer(inp):
  inp = output_enc(inp)
  inp = embedding.to_embed_seq(inp)
  output = net.forward(inp)
  answer = embedding.unembed(output)
  strng = ""
  for i in answer:
    strng = strng + i
  return strng
 
def chat():
  print("Чтобы завершить программу, напишите \"/с(русская с)\", \"/s\", или \"/stop\"")
  while True:
    inp = input("Вы: ")
    if inp == "/s" or inp=="/с" or inp == "/stop":
      break
    x = answer(inp)
    print("Нейросеть: ", x)

#def telechat(inp)


check_cuda()
load_weights(envi)
chat()

# for i in range(0, len(answ1)-1):
#   quest = quest1[i]
#   answ = answ1[i]
#   prc = round(100/len(answ), 2)
#   print("\nЗапрос: ", quest)
#   ai_answ = answer(quest)
#   start_prc = 0.0
#   for ai_char, corr_char in zip(ai_answ, answ):
#       if ai_char == None:
#         continue
#       if ai_char == corr_char:
#         start_prc += prc
#   if start_prc > 100.00:
#     start_prc = start_prc - (start_prc - 100.00)
    
#   print(f"Ответ нейросети: \"{ai_answ}\"(Ожидаемый ответ: \"{answ.strip()}\", cоответствие: {round(start_prc, 2)}%)")
  



#chat()
###############################################################################################################################
