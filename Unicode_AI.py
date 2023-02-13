#It's just a set of code blanks, don't have any special function(for now).
#Это просто набор заготовок кода, не имеет конкретной функции(пока)'

import pickle

all_symbs = ['A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i', 'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z', 'А', 'а', 'Б', 'б', 'В', 'в', 'Г', 'г', 'Д', 'д', 'Е', 'е', 'Ё', 'ё', 'Ж', 'ж', 'З', 'з', 'И', 'и', 'Й', 'й', 'К', 'к', 'Л', 'л', 'М', 'м', 'Н', 'н', 'О', 'о', 'П', 'п', 'Р', 'р', 'С', 'с', 'Т', 'т', 'У', 'у', 'Ф', 'ф', 'Х', 'х', 'Ц', 'ц', 'Ч', 'ч', 'Ш', 'ш', 'Щ', 'щ', 'Ы', 'ы', 'Ь', 'ь', 'Ъ', 'ъ', 'Э', 'э', 'Ю', 'ю', 'Я', 'я', ':', '-', ',', '.', ';', '?', '!', '(', ')', '"', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ']

def min_max_normalize(array):
    min_val = min(array)
    max_val = max(array)
    normalized_array =[]
    for x in array:
     g = (x - min_val) / (max_val - min_val)
     g = round(g, 3)
     normalized_array.append(g)
    return normalized_array

input_text = input('Enter input: ')
output_text = input('Enter output: ')
input_edit = input_text.split(" ")
output_edit = output_text.split(" ")

def numeraise(input_list, all_list):
  numeraited_list = []
  for num, elem in enumerate(all_list):
    for item in input_list:
      for i in item:
        if i == elem:
          numeraited_list.append(num)
  return numeraited_list

input_saved = numeraise(input_edit, all_symbs)
output_saved = numeraise(output_edit, all_symbs)

print(input_saved)
print(output_saved)
def name_generator(g):
    reqname = "request_" + str(g)
    answname = "answer_"+str(g)
    g+=1
