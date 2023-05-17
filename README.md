## Sum Attentions
Standard Attention Mechanism requirews quadratic complexity depending on the input sequence length. This is major drwaback in Text Summarization Task, which takes long sequences as input values. To mend this problem, sparse attention mechanism has introduced, and the major pretrained model with this sparse attention mechanism is Big Bird and Longformer. This repo uses Two models in two different strategies. And we'll trying to figure out pros and cons of using sparse attention mechanism.

<br>
<br>

## Models

**Transformer XL**

<br>

**Reformer**

<br>

**Longformer**

<br>

**Bigbird**


<br><br>

## Setups

**common configs:** <br>
&emsp; hidden_size = 512 <br>
&emsp; intermediate_size = 2048 <br>
&emsp; num_heads = 8 <br>
&emsp; num_layers = 6 <br><br>


| &emsp; Encoder &emsp; | &emsp; Params &emsp; | &emsp; Vocab Size &emsp; | &emsp; Tokenizer Type &emsp; |
|       ---:       |      ---:       |      ---:     |         ---:       |
| &emsp; **`Transformer XL`** &emsp; | &emsp; 32,404,408 &emsp; | &emsp; 267,735 &emsp;  | &emsp; Word Tokenization &emsp; |
| &emsp; **`Reformer`** &emsp; | &emsp; 18,314,240 &emsp; | &emsp; 320 &emsp; | &emsp; Char Tokenizer &emsp; |
| &emsp; **`Longformer`** &emsp; | &emsp; 68,535,552 &emsp; | &emsp; 30,522 &emsp; | &emsp; Sub Word Tokenization &emsp; |
| &emsp; **`BigBird`** &emsp; | &emsp; 45,486,592 &emsp; | &emsp; 50,358 &emsp; | &emsp; Sub Word Tokenization &emsp; |

<br><br>


## Results

<br><br>

## Reference
&nbsp; [**Attention Is All You Need**](https://arxiv.org/abs/1706.03762) <br>
&nbsp; [**Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context**](https://arxiv.org/abs/1901.02860) <br>
&nbsp; [**Reformer: The Efficient Transformer**](https://arxiv.org/abs/2001.04451) <br>
&nbsp; [**Longformer: The Long-Document Transformer**](https://arxiv.org/abs/2004.05150) <br>
&nbsp; [**Big Bird: Transformers for Longer Sequences**](https://arxiv.org/abs/2007.14062) <br>
