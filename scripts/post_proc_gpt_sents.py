from Levenshtein import ratio
import re
import pandas as pd


def read_file(filename):
    with open(filename, "r") as f:
        lines = f.read()
        read_data = [line.strip() for line in lines.split("\n") if line.strip()]
        #if read_data[0] == "":
        #    read_data.pop(0)
        return read_data
        
        
def write_file(filename, sents):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(sents))
        

def extract_txt_after_kw(text, keyword):
    # Create a regular expression pattern to find the keyword (case insensitive) followed by a colon
    pattern = f"{keyword}:\s*(.*)"
    match = re.search(pattern, text, re.IGNORECASE)

    # If a match is found, return the text after the colon
    if match:
        return True, match.group(1).strip()
    else:
        return False, text


# Define a function to extract the Eingabetext-Ausgabetext pairs
def get_examples_from_prompt(prompt):
    # Split the input text into lines
    lines = prompt.split('\n')
    
    # Initialize an empty list to hold the extracted pairs
    pairs = []
    
    # Temporary variables to hold the current Eingabetext and Ausgabetext
    current_eingabe = ""
    current_ausgabe = ""
    
    # Iterate over each line in the input text
    for line in lines:
        # Check if the line starts with "Eingabetext: "
        if line.startswith("Sisendtekst: "):
            # If there is a current Eingabetext, it means we are moving to a new pair
            # and should save the current pair if Ausgabetext is also set
            if current_eingabe and current_ausgabe:
                pairs.append((current_eingabe, current_ausgabe))
                current_ausgabe = ""  # Reset Ausgabetext for the next pair
            current_eingabe = line[len("Sisendtekst: "):]  # Extract the Eingabetext
        # Check if the line starts with "Ausgabetext: "
        elif line.startswith("Väljundtekst: "):
            current_ausgabe = line[len("Väljundtekst: "):]  # Extract the Ausgabetext
    
    #print(pairs)
    return pairs


def compare_lens(sent_1, sent_2):
    
    sent_1_len = len(sent_1)
    sent_2_len = len(sent_2)
    
    if sent_1_len >= sent_2_len:
        len_sim = sent_2_len/sent_1_len
        
    else:
        len_sim = sent_1_len/sent_2_len
        
    return len_sim
    

def is_in_shots(sent, sent_shots):
    for shot in sent_shots:
        if sent == shot[0] or sent == shot[1]:
            return True
    return False
        

def post_process(src_sents, out_sents, shots):
    
    no_response = 0
    nr_of_len_mismatch_sents = 0
    nr_of_valid_sents = 0
    nr_of_proc_sents_re = 0
    nr_of_proc_sents = 0
    out_sents_proc = []
    out_sents_proc_test = []
    
    for src, out, shot in zip(src_sents, out_sents, shots):
        out_str = str(out).strip()
        
        if '\n' in out_str:
            out_split = out_str.split('\n')
            
            candidates = []
            for line in out_split:
                regex_proc = False
                
                # Väljundtekst = Вихідний текст
                if "Väljundtekst" in line or "väljundtekst" in line:
                    regex_proc, line_str = extract_txt_after_kw(line, "väljundtekst")
                    candidates.append((line_str, ratio(src, line_str)))
                    
                # Sisendtekst = Вхідний текст
                elif "Sisendtekst:" not in line and "sisendtekst:" not in line:
                    line_str = line.strip()
                    candidates.append((line_str, ratio(src, line_str)))
                    
                if regex_proc:
                    nr_of_proc_sents_re += 1
                
                                
            nr_of_proc_sents += 1
                
            sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            
            candidate_found = False
            for candidate in sorted_candidates:
                if candidate[1] > 0.7 and candidate[1] < 1 and compare_lens(src, candidate[0]) > 0.66 and is_in_shots(candidate[0], shot) is False:
                    out_sents_proc.append(candidate[0])
                    candidate_found = True
                    break
                
            if candidate_found is False:
                out_sents_proc.append(src)
                out_sents_proc_test.append((src, out_str)) # CHECK
                
        elif out in ['__flagged__', '__invalidrequest__']:
            out_sents_proc.append(src)
            out_sents_proc_test.append((src, out_str)) # CHECK
            no_response+=1
                
        elif len(src) <= 35 and len(out_str) <= 35:
            out_sents_proc.append(out_str)
        
        elif compare_lens(src, out_str) > 0.66:
            out_sents_proc.append(out_str)
            
        else:
            out_sents_proc.append(src)
            out_sents_proc_test.append((src, out_str)) # CHECK
            nr_of_len_mismatch_sents += 1
            
    print(f"There was noise in {nr_of_proc_sents} sents and {nr_of_proc_sents_re} were processed with regex. {nr_of_len_mismatch_sents} sents had mismatching lengths. {no_response} sents didn't get a response.")
    
    return out_sents_proc, out_sents_proc_test
    
    
### ET ###
#work_dir = 'pretrain/ut-rand-reprompting/ut-rand-reprompt-2/'

work_dir = 'ut-zeroshot-gpt-4-turbo/'
err_df = pd.read_csv(work_dir+'out-merged.csv', usecols= ['index','text'])
err_prompts_df = pd.read_csv('ut-zeroshot/prompts.csv', usecols= ['index','text'])
# corr = read_file('pretrain/et-9k-gold/train.est0_Latn-est_Latn.est_Latn')
corr = read_file('ut-zeroshot-gpt-4-turbo/train.est0_Latn-est_Latn.est_Latn')

err_prompts = err_prompts_df['text'].tolist()
err = err_df['text'].tolist()

shots_pairs = [get_examples_from_prompt(prompt) for prompt in err_prompts]

print("prompts:", len(err_prompts))
print("shots:", len(shots_pairs))
print("err:", len(err))
print("corr:", len(corr))

proc_sents, unmatched = post_process(corr, err, shots_pairs)

proc_sents[len(proc_sents)-1] = proc_sents[len(proc_sents)-1]+'\n'

if len(proc_sents) == len(corr):
    print("Post-proc length matches, writing to file.")
    write_file(work_dir+'out-post-proc.txt', proc_sents)
else:
    print("Didn't write processed sents to file, there was a mismatch in lengths.")
