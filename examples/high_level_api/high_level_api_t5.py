from llama_cpp import Llama

llama = Llama("/mnt/md0/models/t5-base.gguf")
tokens = llama.tokenize(b"translate English to German: The house is wonderful.")
llama.encode(tokens)
tokens = [llama.decoder_start_token()]
for token in llama.generate(tokens, top_k=40, top_p=0.95, temp=1, repeat_penalty=1.0):
    print(llama.detokenize([token]))
    if token == llama.token_eos():
        break
