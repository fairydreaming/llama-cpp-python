import ctypes
import os
import multiprocessing

import llama_cpp

llama_cpp.llama_backend_init(numa=False)

N_THREADS = multiprocessing.cpu_count()
MODEL_PATH = os.environ.get("MODEL", "/mnt/md0/models/t5-base.gguf")

prompt = b"translate English to German: The house is wonderful."

lparams = llama_cpp.llama_model_default_params()
cparams = llama_cpp.llama_context_default_params()
model = llama_cpp.llama_load_model_from_file(MODEL_PATH.encode("utf-8"), lparams)
ctx = llama_cpp.llama_new_context_with_model(model, cparams)

n_past = 0

embd_inp = (llama_cpp.llama_token * (len(prompt) + 1))()

n_of_tok = llama_cpp.llama_tokenize(
    model,
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

batch = llama_cpp.llama_batch_init(n_batch, 0, 1);

# prepare batch for encoding containing the prompt
batch.n_tokens = len(embd_inp)
for i in range(batch.n_tokens):
    batch.token[i] = embd_inp[i];
    batch.pos[i] = i
    batch.n_seq_id[i] = 1
    batch.seq_id[i][0] = 0
    batch.logits[i] = False

llama_cpp.llama_encode(
    ctx,
    batch
)

# now overwrite embd_inp so batch for decoding will initially contain only
# a single token with id acquired from llama_model_decoder_start_token(model)
embd_inp = [llama_cpp.llama_model_decoder_start_token(model)]

while remaining_tokens > 0:
    if len(embd) > 0:

        batch.n_tokens = len(embd)
        for i in range(batch.n_tokens):
            batch.token[i] = embd[i];
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
        logits = llama_cpp.llama_get_logits(ctx)
        n_vocab = llama_cpp.llama_n_vocab(model)

        _arr = (llama_cpp.llama_token_data * n_vocab)(
            *[
                llama_cpp.llama_token_data(token_id, logits[token_id], 0.0)
                for token_id in range(n_vocab)
            ]
        )
        candidates_p = llama_cpp.ctypes.pointer(
            llama_cpp.llama_token_data_array(_arr, len(_arr), False)
        )

        _arr = (llama_cpp.llama_token * len(last_n_tokens_data))(*last_n_tokens_data)
        llama_cpp.llama_sample_repetition_penalties(
            ctx,
            candidates_p,
            _arr,
            last_n_repeat,
            repeat_penalty,
            frequency_penalty,
            presence_penalty,
        )

        llama_cpp.llama_sample_top_k(ctx, candidates_p, 40, 1)
        llama_cpp.llama_sample_top_p(ctx, candidates_p, 0.8, 1)
        llama_cpp.llama_sample_temp(ctx, candidates_p, 0.2)
        id = llama_cpp.llama_sample_token(ctx, candidates_p)

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
                model, llama_cpp.llama_token(id), buffer, size, 0, True
            )
            assert n <= size
            print(
                buffer[:n].decode("utf-8"),
                end="",
                flush=True,
            )

    if len(embd) > 0 and embd[-1] in [llama_cpp.llama_token_eos(model), llama_cpp.llama_token_eot(model)]:
        break

print()

llama_cpp.llama_print_timings(ctx)

llama_cpp.llama_batch_free(batch);

llama_cpp.llama_free(ctx)
