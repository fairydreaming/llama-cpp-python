import ctypes
import os
import multiprocessing

import llama_cpp

llama_cpp.llama_backend_init(numa=False)

N_THREADS = multiprocessing.cpu_count()
MODEL_PATH = os.environ.get("MODEL", "/mnt/md0/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")

prompt = b"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

lparams = llama_cpp.llama_model_default_params()
model = llama_cpp.llama_load_model_from_file(MODEL_PATH.encode("utf-8"), lparams)

vocab = llama_cpp.llama_model_get_vocab(model)

cparams = llama_cpp.llama_context_default_params()
cparams.no_perf = False
ctx = llama_cpp.llama_init_from_model(model, cparams)

sparams = llama_cpp.llama_sampler_chain_default_params()
smpl = llama_cpp.llama_sampler_chain_init(sparams)
llama_cpp.llama_sampler_chain_add(smpl, llama_cpp.llama_sampler_init_greedy())

# determine the required inference memory per token:
tmp = [0, 1, 2, 3]

batch = llama_cpp.llama_batch_init(4, 0, 1)

batch.n_tokens = len(tmp)
for i in range(batch.n_tokens):
    batch.token[i] = tmp[i]
    batch.pos[i] = i
    batch.n_seq_id[i] = 1
    batch.seq_id[i][0] = 0
    batch.logits[i] = False

llama_cpp.llama_decode(
    ctx,
    batch
)

llama_cpp.llama_batch_free(batch)

n_past = 0

embd_inp = (llama_cpp.llama_token * (len(prompt) + 1))()

n_of_tok = llama_cpp.llama_tokenize(
    vocab,
    prompt,
    len(prompt),
    embd_inp,
    len(embd_inp),
    True,
    True,
)

embd_inp = embd_inp[:n_of_tok]

n_ctx = llama_cpp.llama_n_ctx(ctx)

n_predict = 20
n_predict = min(n_predict, n_ctx - len(embd_inp))

input_consumed = 0
input_noecho = False

remaining_tokens = n_predict

embd = []
last_n_size = 64
last_n_tokens_data = [0] * last_n_size
n_batch = 24
last_n_repeat = 64
repeat_penalty = 1
frequency_penalty = 0.0
presence_penalty = 0.0

batch = llama_cpp.llama_batch_init(n_batch, 0, 1)

while remaining_tokens > 0:
    if len(embd) > 0:

        batch.n_tokens = len(embd)
        for i in range(batch.n_tokens):
            batch.token[i] = embd[i]
            batch.pos[i] = n_past + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = i == batch.n_tokens - 1

        llama_cpp.llama_decode(
            ctx,
            batch
        )

    n_past += len(embd)
    embd = []
    if len(embd_inp) <= input_consumed:
        id = llama_cpp.llama_sampler_sample(smpl, ctx, -1)

        last_n_tokens_data = last_n_tokens_data[1:] + [id]
        embd.append(id)
        input_noecho = False
        remaining_tokens -= 1
    else:
        while len(embd_inp) > input_consumed:
            embd.append(embd_inp[input_consumed])
            last_n_tokens_data = last_n_tokens_data[1:] + [embd_inp[input_consumed]]
            input_consumed += 1
            if len(embd) >= n_batch:
                break
    if not input_noecho:
        for id in embd:
            size = 32
            buffer = (ctypes.c_char * size)()
            n = llama_cpp.llama_token_to_piece(
                vocab, llama_cpp.llama_token(id), buffer, size, 0, True
            )
            assert n <= size
            print(
                buffer[:n].decode("utf-8"),
                end="",
                flush=True,
            )

    if len(embd) > 0 and embd[-1] in [llama_cpp.llama_token_eos(vocab), llama_cpp.llama_token_eot(vocab)]:
        break

print()

llama_cpp.llama_perf_context_print(ctx)

llama_cpp.llama_batch_free(batch)

llama_cpp.llama_free(ctx)

llama_cpp.llama_model_free(model)
