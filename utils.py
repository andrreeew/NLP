import jieba
import jieba.posseg as posseg
import numpy as np
from sklearn.metrics import accuracy_score, classification_report


def get_top_words(texts, stop_words, remove=[], top=3000):
    # texts: [text1, text2, ...]
    # stop_words: [word1, word2, ...]
    # remove: ['a', 'n', ...]，需要移除的词的词性

    remove = [item.lower() for item in remove]
    counts = {}
    for text in texts:
        words = posseg.cut(text)
        words = [word for word, flag in words if flag[0].lower() not in remove]
        words = set(words)

        for word in words:
            # if len(word) == 1 or word.isdigit() or word in stop_words:
            if word.isdigit():    
            # if word.isdigit() or word in stop_words:    
                continue
            else:
                counts[word] = counts.get(word,0) + 1
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top_words = [item[0] for item in counts[:top]]
    return top_words


def get_text_encoding(texts, dictionary):
    # texts: [text1, text2, ...]
    # dictionary: [word1, word2, ....]

    encoding_size = len(dictionary)
    text_encoding = np.zeros((len(texts), encoding_size))
    for textId in range(len(texts)):
        words = set(jieba.cut(texts[textId]))
        for word in words:
            for dicId  in range(encoding_size):
                if word == dictionary[dicId]:
                    text_encoding[textId, dicId] = 1
                    break
            
    return text_encoding


def pre_process(texts, stop_words_path, N=3000):
    # texts: [text1, text2, ...]
    # output: [[1, 2, 3], [2, 1, 3], [1]]

    with open(stop_words_path, 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f.readlines()]

    print('停用词：{}'.format(len(stop_words)))

    remove = ['r', 't']
    dictionary = get_top_words(texts, stop_words, remove=remove, top=N)
    print('字典：', dictionary)
    
    text_encoding = get_text_encoding(texts, dictionary)

    return text_encoding, dictionary


def test(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y, y_pred))



def assemble_test(models, X, y):
    num = len(models)
    y_pred = np.zeros(len(y))
    for model in models:
        y_pred += np.array(model.predict(X))

    y_pred = [1 if value >= num*1.0/2 else 0 for value in y_pred]

    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y, y_pred))