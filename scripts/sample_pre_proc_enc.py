from estnltk import Text
import random
import time
import string

punct = ["'", '´', "'", '"', '"', '«', '„', '»']
punct.extend(string.punctuation)

def filter_sents(sents):
    filtered = dict()
    for sent,index in sents.items():
        if sent[0].isalpha():
            if sent[0].isupper() and sent[len(sent)-1] in punct:
                filtered[sent] = index
        elif sent[0] in ["'", '´', "'", '"', '"', '«', '„', '»']:
            if sent[len(sent)-1] in punct:
                filtered[sent] = index
        elif sent[0].isnumeric():
            if sent[len(sent)-1] in punct:
                filtered[sent] = index
    return filtered

def read_data_lines(filename, indices):
    data = dict()
    fp = open(filename, "r", encoding="utf-8")
    for i, line in enumerate(fp):
        if i in indices.keys():
            line_str = line.strip()
            line_text = Text(line_str)
            line_text.tag_layer()
            if len(line_text.words) > 3 and len(line_text.words) < 70:
                if line_str not in data.keys():
                    data[line_str] = i
    fp.close()
    return data

def write_file(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(data))
        
filenames = {"koondkorpus/ENC_prevert_sentences/no_tags_nc21_Wikipedia_2021.prevert": [705215, 'koondkorpus_10k/no_tags_nc21_Wikipedia_2021_rand_subset.txt', 'koondkorpus_10k/no_tags_nc21_Wikipedia_2021_rand_subset_indices.txt'],
            "koondkorpus/ENC_prevert_sentences/no_tags_nc21_Web_2021.prevert": [62244869, 'koondkorpus_10k/no_tags_nc21_Web_2021_rand_subset.txt', 'koondkorpus_10k/no_tags_nc21_Web_2021_rand_subset_indices.txt'],
            "koondkorpus/ENC_prevert_sentences/no_tags_nc21_Fiction.prevert": [1504214, 'koondkorpus_10k/no_tags_nc21_Fiction_rand_subset.txt', 'koondkorpus_10k/no_tags_nc21_Fiction_rand_subset_indices.txt']}

nr_of_sents = 10000

for filename, info in filenames.items():
    start_time = time.time()
    rand_indices = random.sample(range(0, info[0]), nr_of_sents)
    indices = {key: None for key in rand_indices}
    rand_lines = read_data_lines(filename, indices)
    filtered_lines = filter_sents(rand_lines)

    final_sents = list(filtered_lines.keys())
    final_indices = [str(index) for index in filtered_lines.values()]
    write_file(info[1], final_sents)
    write_file(info[2], final_indices)

    end_time = time.time()
    print(filename, len(filtered_lines), "time:", end_time-start_time)
