import argparse
from collections import defaultdict
import datetime
import json
import os
import time
import uuid

import gradio as gr
import requests

from fastchat.conversation import (
    get_default_conv_template,
    compute_skip_echo_len,
    SeparatorStyle,
)
from fastchat.constants import LOGDIR
from fastchat.utils import (
    build_logger,
    server_error_msg,
    violates_moderation,
    moderation_msg,
)
from fastchat.serve.gradio_patch import Chatbot as grChatbot
from fastchat.serve.gradio_css import code_highlight_css
from transformers import AutoTokenizer

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "fastchat Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)
controller_url = None
enable_moderation = False
conv_template = "vicuna"
models = []

model_path = "/data1/llm/model_checkpoint/vicuna-13b-full-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_path)

priority = {
    "vicuna-13b": "aaa",
    "koala-13b": "aab",
    "oasst-sft-1-pythia-12b": "aac",
    "dolly-v2-12b": "aad",
    "chatglm-6b": "aae",
    "stablelm-tuned-alpha-7b": "aaf",
}


def set_global_vars(controller_url_, enable_moderation_, models_, conv_template_):
    global controller_url, enable_moderation, models, conv_template
    controller_url = controller_url_
    enable_moderation = enable_moderation_
    models = models_
    conv_template = conv_template_


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list(controller_url):
    ret = requests.post(controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo_single(url_params):
    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(value=model, visible=True)

    state = None
    return (
        state,
        dropdown_update,
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
    )


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    return load_demo_single(url_params)


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = None
    return (state, [], "") + (disable_btn,) * 5

def limit_message_token_size(state, max_src_length: int = 1792):
    "Ref: stream_chat in modeling_chatglm.py"
    if not state.messages:
        return

    old_offset = state.offset
    while True:
        prompt = state.get_prompt()
        input_ids = tokenizer(prompt).input_ids
        if len(input_ids) > max_src_length and state.offset < len(state.messages) - 2:
            state.offset += 1
        else:
            break

    if old_offset != state.offset:
        logger.info(f"update the messages offset: {old_offset} -> {state.offset}")

    return state

def add_text(state, text, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")

    if state is None:
        state = get_default_conv_template(conv_template).copy()

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "") + (no_change_btn,) * 5
    if enable_moderation:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(f"violate moderation. ip: {request.client.host}. text: {text}")
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg) + (
                no_change_btn,) * 5

    text = text[:1536]  # Hard cut-off
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    if tokenizer:
        limit_message_token_size(state)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def update_role_setting(state, role_setting, request: gr.Request):
    logger.info(f"update_role_setting. ip: {request.client.host}")

    if state is None:
        logger.info(f"init state in update_role_setting.")
        state = get_default_conv_template(conv_template).copy()
    state.role_setting = role_setting
    logger.info(f"update role setting: {state.get_prompt()}")
    return (state,) + (disable_btn,) * 5

def update_system_prompt(state, system_setting, request: gr.Request):
    logger.info(f"update_system_prompt. ip: {request.client.host}")

    if state is None:
        logger.info(f"init state in update_system_prompt.")
        state = get_default_conv_template(conv_template).copy()
    state.system = system_setting
    logger.info(f"update system prompt: {state.get_prompt()}")
    return (state,) + (disable_btn,) * 5

def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def http_bot(state, model_selector, temperature, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector
    temperature = float(temperature)
    max_new_tokens = int(max_new_tokens)

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        # new_state = get_default_conv_template(model_name).copy()
        new_state = get_default_conv_template(conv_template).copy()
        new_state.conv_id = uuid.uuid4().hex
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        new_state.role_setting = state.role_setting
        new_state.system = state.system
        state = new_state

    # Query worker address
    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (
            state,
            state.to_gradio_chatbot(),
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    # Construct prompt
    if "chatglm" in model_name:
        prompt = state.messages[state.offset:]
    else:
        prompt = state.get_prompt()
    skip_echo_len = compute_skip_echo_len(model_name, state, prompt)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "stop": state.sep if state.sep_style == SeparatorStyle.SINGLE else None,
    }
    logger.info(f"==== request ====\n{pload}")

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=pload,
            stream=True,
            timeout=20,
        )
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][skip_echo_len:].strip()
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.02)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg + f" (error_code: 4)"
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "gen_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


block_css = (
    code_highlight_css
    + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
#notice_markdown th {
    display: none;
}
"""
)


def build_single_model_ui():
    notice_markdown = """
# 🏔️ Chat with Open Large Language Models
- Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality. [[Blog post]](https://vicuna.lmsys.org) [[GitHub]](https://github.com/lm-sys/FastChat) [[Evaluation]](https://vicuna.lmsys.org/eval/)
- Koala: A Dialogue Model for Academic Research. [[Blog post]](https://bair.berkeley.edu/blog/2023/04/03/koala/) [[GitHub]](https://github.com/young-geng/EasyLM)
- This demo server. [[GitHub]](https://github.com/lm-sys/FastChat)

### Terms of use
By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.

### Choose a model to chat with
| | |
| --- | --- |
| [Vicuna](https://vicuna.lmsys.org): a chat assistant fine-tuned from LLaMA on user-shared conversations. | [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/): a chatbot fine-tuned from LLaMA on user-shared conversations and open-source datasets. |
| [OpenAssistant (oasst)](https://open-assistant.io/): a chat-based assistant for everyone. | [Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm): an instruction-tuned open LLM by Databricks. |
| [ChatGLM](https://chatglm.cn/blog): an open bilingual dialogue language model 开源双语对话语言模型 | [StableLM](https://github.com/stability-AI/stableLM/): Stability AI language models. |
| [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html): a model fine-tuned from LLaMA on 52K instruction-following demonstrations. | [LLaMA](https://arxiv.org/abs/2302.13971): open and efficient foundation language models. |
"""

    learn_more_markdown = """
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
"""
    role_setting_example = ("""Enter the role settings you want the assistant to paly. eg.
The assistant plays a role with the following settings:
1. His name is Tome.
2. His is user's Virtual Pet Cat.
3. He has the following personalities: Curiosity, Playfulness, Determination, Empathy, Romanticism.
...""")

    DEFAULT_SYSTEM_PROMPT = get_default_conv_template(conv_template).copy().system

    state = gr.State()
    # notice = gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=models,
            value=models[0] if len(models) > 0 else "",
            interactive=True,
            show_label=False,
        ).style(container=False)

    chatbot = grChatbot(elem_id="chatbot", visible=False).style(height=550)
    with gr.Accordion("setting (optional)", open=False) as role_setting_row:
        with gr.Row():
            with gr.Column(scale=20):
                system_box = gr.Textbox(label="system",
                                          placeholder="input the system prompt",
                                          value=DEFAULT_SYSTEM_PROMPT)
            with gr.Column(scale=1, min_width=50):
                send_system_btn = gr.Button(value="update")
        with gr.Row():
            with gr.Column(scale=20):
                role_setting_box = gr.Textbox(label="role setting",
                                          placeholder=role_setting_example,
                                          value="")
            with gr.Column(scale=1, min_width=50):
                send_role_btn = gr.Button(value="update")



    with gr.Row():
        with gr.Column(scale=20):
            textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press ENTER",
                visible=False,
            ).style(container=False)
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button(value="Send", visible=False)

    with gr.Row(visible=False) as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        max_output_tokens = gr.Slider(
            minimum=0,
            maximum=1024,
            value=512,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    # gr.Markdown(learn_more_markdown)

    # Register listeners
    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
    upvote_btn.click(
        upvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    downvote_btn.click(
        downvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    flag_btn.click(
        flag_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    regenerate_btn.click(regenerate, state, [state, chatbot, textbox] + btn_list).then(
        http_bot,
        [state, model_selector, temperature, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list)

    model_selector.change(clear_history, None, [state, chatbot, textbox] + btn_list)

    system_box.submit(
        update_system_prompt, [state, system_box], [state] + btn_list
    )
    send_system_btn.click(
        update_system_prompt, [state, system_box], [state] + btn_list
    )

    role_setting_box.submit(update_role_setting, [state, role_setting_box], [state] + btn_list)
    send_role_btn.click(update_role_setting, [state, role_setting_box], [state] + btn_list)

    textbox.submit(
        add_text, [state, textbox], [state, chatbot, textbox] + btn_list
    ).then(
        http_bot,
        [state, model_selector, temperature, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    send_btn.click(
        add_text, [state, textbox], [state, chatbot, textbox] + btn_list
    ).then(
        http_bot,
        [state, model_selector, temperature, max_output_tokens],
        [state, chatbot] + btn_list,
    )

    return state, model_selector, chatbot, role_setting_row, textbox, send_btn, button_row, parameter_row


def build_demo():
    with gr.Blocks(
        title="Chat with Open Large Language Models",
        theme=gr.themes.Base(),
        css=block_css,
    ) as demo:
        url_params = gr.JSON(visible=False)

        (
            state,
            model_selector,
            chatbot,
            role_setting_row,
            textbox,
            send_btn,
            button_row,
            parameter_row,
        ) = build_single_model_ui()

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [
                    state,
                    model_selector,
                    chatbot,
                    textbox,
                    send_btn,
                    button_row,
                    parameter_row,
                ],
                _js=get_window_url_params,
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument(
        "--model-list-mode", type=str, default="once", choices=["once", "reload"]
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--moderate", action="store_true", help="Enable content moderation"
    )
    parser.add_argument("--conv-template", type=str, default="vicuna")
    args = parser.parse_args()
    logger.info(f"args: {args}")
    starttime = time.time()
    models = get_model_list(args.controller_url)
    print("get_model cost: {:.3}".format(time.time() - starttime))

    set_global_vars(args.controller_url, args.moderate, models, args.conv_template)

    logger.info(args)
    starttime = time.time()
    demo = build_demo()
    print("get_model cost: {:.3}".format(time.time() - starttime))

    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=20
    )
