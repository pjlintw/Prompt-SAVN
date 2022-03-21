import json
from sa_utils import get_slots, read_multiwoz_examples, convert_examples_to_features

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import torch.nn as nn

from transformers import (
    WEIGHTS_NAME, 
    BertConfig, 
    BertTokenizer, 
    AdamW, 
    get_linear_schedule_with_warmup)

def read_multiwoz_examples_2(input_file, is_training, data_version):
    slots = get_slots(data_version)

    input_file = "data/2.1/train_dials.json"
    with open(input_file, "r", encoding="utf-8") as reader:
        data = json.load(reader)
    
    for entry in data:
        dialogues = entry["dialogue"]
        print(dialogues)

        for i, turn in enumerate(dialogues):
            print(turn)
        break

    print(slots)
    return slots


def convert_id_to_value_type(inp):
    """Maaping token id into value type.

    The token ids of `none`, `no` and `span` will be
    convert into value type {0,1,2,3}. 
    0 is for `none` and ``

    value type   token ids (pattern)    token_ids (no pattern)   
    `none`     ==  `3904`            == `0`
    `dontcare` ==  `2053`            == `1`
    `span`     ==  `8487`            == `2`
    ``         == any other token id == `3`
    Args:
      inp: 1D tensor
    """
    # assert inp[0] == int
    #assert torch.is_tensor(inp[0]) == False
    MAPPING = {3904: 0, 2053:1, 8487:2}
    return [MAPPING[v] if v in MAPPING else 3 for v in inp]

    
def main(): 
    t = torch.tensor([1,2,3904, 2053, 8487], dtype=torch.long)
    new_t = convert_id_to_value_type(t)
    print(t)
    print(new_t)
    return 

    n_tokens=5
    ids = torch.full((n_tokens,), 50256)
    x = torch.tensor([50256, 50256])
    print(ids)
    print(x)
    tok = tokenizer.convert_ids_to_tokens(x)
    print(tok)

    tokenizer_name = "model/bert-base-savn-vocab.txt"
    model_name_or_path = "bert-base-uncased"
    if tokenizer_name:
        print("using: ", tokenizer_name)
    else:
        print("using", model_name_or_path)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name if tokenizer_name else model_name_or_path,
                                              do_lower_case=True)
    

    examples = read_multiwoz_examples("data/2.1/train_dials.json", True, "2.1")
    
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=512,
                                            is_training=True,
                                            use_sp=True)
    print(len(examples))
    
    # for i in range(0):
    #     print(examples[i].dialogue_idx)
    #     print(examples[i].domains)
    #     print(examples[i].turn_idx)
    #     print(examples[i].doc_tokens)
    #     print(examples[i].orig_value_text)
    #     print(examples[i].example_domains)
    #     print()

    #     #M UL2429.json
    #     # all domains in the whole dialogue ['restaurant', 'taxi', 'attraction']
    #     # 6
    #     # List of tokens including [USER],[SYS],[SEP]
    #     # List of value for 30 slots: ['dontcare', 'none', 'museum', 'none'...]
    #     # ['attraction', 'restaurant', 'taxi']
    

    # for i in range(0):
    #     print("unique_id", features[i].unique_id)
    #     print("dialogue_idx", features[i].dialogue_idx)
    #     print("turn_idx", features[i].turn_idx)
    #     print("raw_tokens", features[i].raw_tokens)
    #     print("tok_to_orig_index", features[i].tok_to_orig_index)
    #     print("all_doc_tokens", features[i].all_doc_tokens)
    
    #     # Required for training
    #     print("input_ids", features[i].input_ids)
    #     print("attention_masks", features[i].attention_masks)
    #     print("token_type_ids", features[i].token_type_ids)
    #     print("value_types", features[i].value_types)
    #     print("start_positions", features[i].start_positions)
    #     print("end_positions", features[i].end_positions)
    #     print("domains", features[i].domains) 
    #     print()
        #     MultiWozFeatures(
        # unique_id=unique_id,
        # dialogue_idx=example.dialogue_idx,
        # turn_idx=example.turn_idx,
        # raw_tokens=raw_tokens,
        # tok_to_orig_index=tok_to_orig_index,
        # all_doc_tokens=all_doc_tokens,
        # input_ids=input_ids,
        # attention_masks=attention_masks,
        # token_type_ids=token_type_ids,
        # value_types=value_types,
        # start_positions=start_positions,
        # end_positions=end_positions,
        # domains=domains


if __name__ == "__main__":
    main()


