import pickle
import os


def encode(input, search_list):
  encode_list = []
  for i in input:
    for num, item in enumerate(search_list):
      if item == i:
        encode_list.append(num)
      else:
        continue
  return encode_list

def decode(input, search_list):
  decode_list = []
  for i in input:
    for num, item in enumerate(search_list):
        if i == num:
          decode_list.append(item)
        else:
          continue
  return decode_list

if os.path.isfile('import_list.pickle'):
      print('Exist')
else:
      with open('import_list.pickle', "xb") as f:
        print('Created')
        
input_text = input('Enter some text: ')
input_list = input_text.split(" ")
print(input_list)
all_list = []
    
with open('import_list.pickle', "rb") as f:
  try:
      all_list = pickle.load(f)
  except EOFError:
      print("nothing in file")
   
with open('import_list.pickle', "wb") as f:
    for item in input_list:
        if item in all_list:
          print("in all list")
          continue
        else:          
          all_list.append(item)
    pickle.dump(all_list, f)        

def redact(input):
  redacted_list = []
  punc = ', . ? - \" \' : ; ! *'
  for item in input:
    new_item = ""
    for i in item:
      if i not in punc:
        new_item += i
    redacted_list.append(new_item.lower())
  return redacted_list


print(all_list)
encoded_list = encode(input_list, all_list)
decoded_list = decode(encoded_list, all_list)
print(encoded_list)
print(decoded_list)
list = redact(input_list)
print(list)

def inunicode(input):
  unicoded_input_list = []
  for char in input:
    x = int(ord(char))
    unicoded_input_list.append(x)
  return unicoded_input_list
  
def ununicode(input):
  text = ""
  for num in input:
    text += chr(int(num))
  return text

inuni = inunicode(input_text)
ununi = ununicode(inuni)
print(inuni)
print(ununi)