"""
## Minería de textos
Universidad de Alicante, curso 2020-2021

Esta documentación forma parte de la práctica "[Lectura y documentación de un sistema de
extracción de entidades](https://jaspock.github.io/mtextos/bloque2_practica.html)" y se 
basa en el código del curso [CS230](https://github.com/cs230-stanford/cs230-code-examples) 
de la Universidad de Stanford.

**Autor de los comentarios:** Nombre Completo

Este módulo crea los siguientes ficheros a partir del contenido del directorio indicado 
como argumento: `words.txt`, que contiene...; `tags.txt`, que contiene...
"""

import argparse
from collections import Counter
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=1, help="Minimum count for words in the dataset", type=int)
parser.add_argument('--min_count_tag', default=1, help="Minimum count for tags in the dataset", type=int)
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")

PAD_WORD = '<pad>'
PAD_TAG = 'O'
UNK_WORD = 'UNK'

"""
## Título de sección de nivel 2

### Título de sección de nivel 3
"""

def save_vocab_to_txt_file(vocab, txt_path):
    """
    ### Función `save_vocab_to_txt_file`
    Recorre todas las palabras del diccionario (keys) y escribe todas las palabras, 1 por línea
    #### Parámetros:

    * `vocab`: vocabulario que contiene todas las palabras del corpus
    * `txt_path`: ruta del fichero a escribir con formato .txt
    """
    with open(txt_path, "w") as f:  # Este tipo de comentario es ignorado por pycco.
        for token in vocab:
            f.write(token + '\n')
            

def save_dict_to_json(d, json_path):
    """
    ### Función `save_dict_to_json`
    Escribe el diccionario de entrada en un fichero con formato json.

    #### Parámetros:

    * `d`: diccionario con estadísticas sobre datos de train/val/test y el vocabulario.
    * `json_path`: ruta del json del fichero a escribir
    """

    with open(json_path, 'w') as f:
        # Los comentarios cortos aparecen en la documentación si no hay código en esa línea.
        # En general, la mayoría de los comentarios serán largos y usarán docstrings.
        d = {k: v for k, v in d.items()}
        """
        Este es un comentario más largo. La variable `d` es un diccionario que contiene estadísticas sobre los datos
        y sobre el vocabulario del corpus.
        """
        json.dump(d, f, indent=4)


def update_vocab(txt_path, vocab):
    """
    ### Función `update_vocab`
    Construye y actualiza el vocabulario leyendo las líneas del fichero de entrada.
    Devuelve el número de líneas totales
    #### Parámetros:

    * `txt_path`: ruta del fichero de texto a leer
    * `vocab`: vocabulario con las palabras del corpus que lleva el conteo.  
    """

    with open(txt_path) as f:
        for i, line in enumerate(f):
            vocab.update(line.strip().split(' '))

    return i + 1


if __name__ == '__main__':

    args = parser.parse_args()

    print("Building word vocabulary...")
    words = Counter()
    size_train_sentences = update_vocab(os.path.join(args.data_dir, 'train/sentences.txt'), words)
    size_dev_sentences = update_vocab(os.path.join(args.data_dir, 'val/sentences.txt'), words)
    size_test_sentences = update_vocab(os.path.join(args.data_dir, 'test/sentences.txt'), words)
    print("- done.")

    print("Building tag vocabulary...")
    tags = Counter()
    size_train_tags = update_vocab(os.path.join(args.data_dir, 'train/labels.txt'), tags)
    size_dev_tags = update_vocab(os.path.join(args.data_dir, 'val/labels.txt'), tags)
    size_test_tags = update_vocab(os.path.join(args.data_dir, 'test/labels.txt'), tags)
    print("- done.")

    assert size_train_sentences == size_train_tags
    assert size_dev_sentences == size_dev_tags
    assert size_test_sentences == size_test_tags

    words = [tok for tok, count in words.items() if count >= args.min_count_word]
    tags = [tok for tok, count in tags.items() if count >= args.min_count_tag]

    if PAD_WORD not in words: words.append(PAD_WORD)
    if PAD_TAG not in tags: tags.append(PAD_TAG)
    
    words.append(UNK_WORD)

    print("Saving vocabularies to file...")
    save_vocab_to_txt_file(words, os.path.join(args.data_dir, 'words.txt'))
    save_vocab_to_txt_file(tags, os.path.join(args.data_dir, 'tags.txt'))
    print("- done.")

    # Estadísticas acerca de los splits, dataset y tokens especiales para el entrenamiento
    sizes = {
        'train_size': size_train_sentences,
        'dev_size': size_dev_sentences,
        'test_size': size_test_sentences,
        'vocab_size': len(words),
        'number_of_tags': len(tags),
        'pad_word': PAD_WORD,
        'pad_tag': PAD_TAG,
        'unk_word': UNK_WORD
    }
    save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params.json'))

    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))
