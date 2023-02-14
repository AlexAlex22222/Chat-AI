all_symbs = ['A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i', 'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z', 'А', 'а', 'Б', 'б', 'В', 'в', 'Г', 'г', 'Д', 'д', 'Е', 'е', 'Ё', 'ё', 'Ж', 'ж', 'З', 'з', 'И', 'и', 'Й', 'й', 'К', 'к', 'Л', 'л', 'М', 'м', 'Н', 'н', 'О', 'о', 'П', 'п', 'Р', 'р', 'С', 'с', 'Т', 'т', 'У', 'у', 'Ф', 'ф', 'Х', 'х', 'Ц', 'ц', 'Ч', 'ч', 'Ш', 'ш', 'Щ', 'щ', 'Ы', 'ы', 'Ь', 'ь', 'Ъ', 'ъ', 'Э', 'э', 'Ю', 'ю', 'Я', 'я', ':', '-', ',', '.', ';', '?', '!', '(', ')', '"', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', " "]

input_text = input('Enter input: ')
#output_text = input('Enter output: ')
#output_edit = output_text.split(" ")

def numeraise(input_list, all_list):
    numeraited_dict = {elem:num for num, elem in enumerate(all_list)}
    numeraited_list = [numeraited_dict[i] for item in input_list for i in item]
    return numeraited_list


input_saved = numeraise(input_text, all_symbs)
#output_saved = numeraise(output_edit, all_symbs)

print(input_saved)
#print(output_saved)

def unnumeraise(input_list, all_list):
    unnumeraited_dict = {num:elem for num, elem in enumerate(all_list)}
    unnumeraited_list = [unnumeraited_dict[item] for item in input_list]
    return unnumeraited_list
  
unnum = unnumeraise(input_saved, all_symbs)

print(unnum)