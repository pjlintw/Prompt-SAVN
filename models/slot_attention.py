"""Implemtation of Slot Attention"""
import math
import torch
import torch.nn as nn 
import torch.nn.functional as F
from transformers import BertForMaskedLM,BertPreTrainedModel, BertModel


class AttentionBlock(nn.Module):
    """Attention block which includes self-attention and feed-forward network."""
    def __init__(self, config):
        super(AttentionBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dropout_rate = config.hidden_dropout_prob

        # Attention
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_output_layer = nn.Linear(self.hidden_size, self.hidden_size)

        # FFN 
        self.intermediate_layer = nn.Linear(self.hidden_size, self.intermediate_size)
        self.ffn_output_layer = nn.Linear(self.intermediate_size, self.hidden_size)

        self.attention_dropout = nn.Dropout(self.dropout_rate)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)                
        self.layernorm1 = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(self.hidden_size, eps=1e-6)


    def scaled_dot_product_attention(self, utterance_hidden, slot_hidden, attention_mask):
        """Compute attention."""
        # Linear transform
        q = self.w_q(slot_hidden)
        k = self.w_k(utterance_hidden)
        v = self.w_v(utterance_hidden)
        # Unnormalized attention scores
        attn_score = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(self.hidden_size)
        # Apply mask
        attn_score = attn_score + (1.0 - attention_mask[:, None, :]) * -10000.0
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = self.attention_dropout(attn_weight)

        attn_output = torch.matmul(attn_weight, v)
        attn_output = self.attn_output_layer(attn_output)
        
        return attn_output, attn_weight

    def feed_forward_network(self, inputs):
        """Perform Feed-Forward network."""
        mlp_out = F.gelu(self.intermediate_layer(inputs))
        mlp_out = self.ffn_output_layer(mlp_out)
        return mlp_out

    def forward(self, utterance_hidden, slot_hidden, attention_mask):
        """Perform Slot attetion."""
        # Slot attention
        attn_out, attn_weight = self.scaled_dot_product_attention(utterance_hidden, slot_hidden, attention_mask)
        # Dropout and layer norm
        out = self.dropout1(attn_out)
        out = out + slot_hidden
        out = self.layernorm1(out)
        # FFN
        ffn_out = self.feed_forward_network(out)
        # Dropout and layer norm
        ffn_out = self.dropout2(ffn_out)
        ffn_out = ffn_out + out
        ffn_out = self.layernorm2(ffn_out)

        return ffn_out, attn_weight


class SlotAttentionNetwork(BertPreTrainedModel):
    def __init__(self, config, args):
        super(SlotAttentionNetwork, self).__init__(config)
        # Config
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dropout_rate = config.hidden_dropout_prob
        # Args
        self.cls_lambda = args.cls_lambda
        self.position_lambda = args.ans_lambda
        # Models
        self.bert = BertModel(config)
        
        # Models and Output layers
        # Use MLM cls layer as slot gate layer
        # this results in the output (batch_size, seq_len, 30522)
        if args.use_pattern:
            self.mlm = BertForMaskedLM.from_pretrained(args.model_name_or_path,
                                                       from_tf=bool('.ckpt' in args.model_name_or_path),
                                                       config=config,
                                                       cache_dir=args.cache_dir if args.cache_dir else None,
                                                       )
            self.bert = self.mlm.bert
            self.slot_gate_layer = self.mlm.cls
        else:
            self.bert = BertModel(config)
            self.slot_gate_layer = nn.Linear(self.hidden_size, self.num_labels)    

        # Slot attention
        self.slot_attention_layer = AttentionBlock(config)


        self.start_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.end_layer = nn.Linear(self.hidden_size, self.hidden_size)
        # init weight
        self.init_weights()


    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                value_types=None,
                slot_input_ids=None,
                start_positions=None,
                end_positions=None,
                use_pattern=None):
        """Perform Slot gate classifcation and span prediction.
        
        Args:
            # Get features and labels
            input_ids = inputs["input_ids"]            # (batch_size, seq_len) 
            attention_mask = inputs["attention_mask"]  # (batch_size, seq_len) 
            token_type_ids = inputs["token_type_ids"]  # (batch_size, seq_len) 
            value_types = inputs["value_types"]        # (batch_size, 30): value type is 0 (none), 1 (dontcare) or 2 (span).
            slot_input_ids = inputs["slot_input_ids"]  # slot_input_ids has shape (30, 5)
            start_positions =inputs["start_positions"] # (batch_size, 30): position index or -1
            end_positions = inputs["end_positions"]

        """
        # `utterance_hidden` has shape (batch_size, seq_len, hidden_size)
        # `pool_output`: has shape (batch_size, hidden_size)
        utterance_hidden, pool_output = self.bert(input_ids,
                                                  attention_mask=attention_mask,
                                                  token_type_ids=token_type_ids,
                                                  return_dict=False)
        # Encode slots: (num_slot, max_tok_len) == (30, 5) to (30,768)
        # [`slot_input_ids`, `slot_input_mask`] 
        # `slot_input_ids`: (n_gpu, batch_size, 30, 5)
        # slot_input_ids[0][0]:             (30,5)
        # embeddings(slot_input_ids[0][0]): (30,5,768)  
        # MeanPooling: (30, 5, emb_dims) -> (30, 768)
        slot_hidden = self.bert.embeddings(slot_input_ids[0][0]).mean(-2)        

        # Get first [MASK] token
        # slot_hidden = self.bert.encoder.layer[0](self.bert.embeddings(slot_input_ids[0][0]))[0][:,0,:].squeeze(1)

        # print(slot_hidden)
        # print(type(slot_hidden))
        # print(slot_hidden.shape)
        # return
    
        ### 1. Slot Gate Classification ###
        # Slot attention output has shape (batch_size, num_slot=30, 768)
        slot_attn_out, slot_attn_weight = self.slot_attention_layer(utterance_hidden, slot_hidden, attention_mask) 
        
        # if `args.use_pattern`, the output is (batch_size, 30, 30522)
        # otherwise the shape is (batch_size, 30, 3)
        slot_logits = self.slot_gate_layer(slot_attn_out)

        # print("\nslot hidden before mean", self.bert.embeddings(slot_input_ids[0][0]).shape) # (30,5,768)
        # print("slot hidden", slot_hidden.shape) # (30,768)
        # print("slot_attn_out shape", slot_attn_out.shape)
        # print("slot_logits shape", slot_logits.shape)
        # print("taking argmix", slot_logits.argmax(dim=-1).shape)
        # print(type(slot_logits.argmax(dim=-1)))
        # return 

        ### 2. Span prediction ###
        # Linear transformer
        start_hidden = self.start_layer(slot_hidden)
        end_hidden = self.end_layer(slot_hidden)
        # Multiply with sequence output
        # (batch_size, num_slot, hidden_size) * (batch_size, hidden_size, seq_len)
        # -> (batch_size, num_slot, seq_len)
        start_logits = torch.matmul(start_hidden, utterance_hidden.permute(0,2,1))
        end_logits = torch.matmul(end_hidden, utterance_hidden.permute(0,2,1))

        return (slot_logits, start_logits, end_logits)