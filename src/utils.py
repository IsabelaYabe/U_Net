import os
import shutil

def dict_labels(df_labels):
    rgb_to_index = {}
    index_to_label = {}
    for count, row in df_labels.iterrows():
        color = (row['r'], row['g'], row['b'])
        rgb_to_index[color] = count
        index_to_label[count] = color
    return rgb_to_index, index_to_label

# Função para remover todos subdiretórios e arquivos de um diretório
def remove_all_files_from_dir(dir_path):
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)