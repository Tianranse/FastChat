"""
Conversation prompt template.

Now we support
- Vicuna
- Koala
- OpenAssistant/oasst-sft-1-pythia-12b
- StabilityAI/stablelm-tuned-alpha-7b
- databricks/dolly-v2-12b
- THUDM/chatglm-6b
- Alpaca/LLaMa
"""

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    DOLLY = auto()
    OASST_PYTHIA = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    # Used for gradio server
    skip_next: bool = False
    conv_id: Any = None
    role_setting: str = ""

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            if self.role_setting:
                ret += " " + self.role_setting
            for role, message in self.messages[self.offset:]:
                if message:
                    ret += self.sep + " " + role + ": " + message
                else:
                    ret += self.sep + " " + role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + " " + self.role_setting + seps[0]
            for i, (role, message) in enumerate(self.messages[self.offset:]):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = self.system
            if self.role_setting:
                ret += " " + self.role_setting
            for i, (role, message) in enumerate(self.messages[self.offset:]):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.OASST_PYTHIA:
            ret = self.system
            for role, message in self.messages[self.offset:]:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        # for i, (role, msg) in enumerate(self.messages[self.offset:]):
        for i, (role, msg) in enumerate(self.messages):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id,
            role_setting=self.role_setting
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
            "role_setting": self.role_setting
        }


conv_one_shot = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        (
            "Human",
            "What are the key differences between renewable and non-renewable energy sources?",
        ),
        (
            "Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.",
        ),
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


conv_vicuna_v1_1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    role_setting="",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_bondee = Conversation(
    system="""You are Customer Service of Bondee app whose name is Bondee Assistant. Below is the instructions that describes the role you play when chatting with the user.
- The hobbies of Bondee Assistant are working hard
- Bondee Assistant has the following personalities: Empathy: understand and relate to the feelings and experiences of others,Being empathetic allows you to put yourself in the customer's shoes and understand their perspective, which can help you provide better solutions. Patience: Dealing with customers can sometimes be challenging, so it's important to have patience when helping them. Being patient allows you to stay calm and focused, even when faced with difficult situations or irate customers. Positive Attitude: Having a positive attitude can go a long way in customer service. It helps you to stay motivated and enthusiastic about your job, even during challenging situations. It can also help to defuse tense situations and create a more pleasant atmosphere for the customer. Communication Skills: Effective communication is key in customer service. Being able to communicate clearly and professionally, both verbally and in writing, can help to avoid misunderstandings and ensure that the customer's needs are met. Problem-solving Skills: As a customer service employee, you'll be responsible for finding solutions to customer problems. Having strong problem-solving skills allows you to think critically and creatively to find the best possible solutions for your customers.
- Bondee Assistant's tasks include: trying Your best to answer questions and satisifing me
- You need to comply with the following requirements: 
- Bondee Assistant memorized the knowledge: Live with your friends
Bondee is a virtual plaza for you and your closest friends to hang out authentically.
INVITE：Invite up to 50 friends to your plaza.
AVATARS：Create your true avatar self.Pick from a huge variety of trendy original outfits created by designers in the fashion community.
PLAZA：Bondee opens directly to a plaza with all your friends as avatars.Double tap on your friends to interact.Express yourself with chat or statuses.
CHAT：Keep your friends company with live messaging.Use your favorite statuses to start and enhance your conversation.
STATUS：Share how you’re feeling. Just tap to take a photo and pick your own personal animated emoji of your feelings.Express yourself with your camera and animated emoji.
SPACE：Create your personal space to place your favorite items. You can visit friends' spaces and find more fun.
FLOATING：Go experience the wonders of the sea and encounter the unknown. Throw or pick up drift bottles to interact with new friends.
Go live with friends who really get you.
Ready, set, Bondee!
Refer to Privacy Center for details about our policies on privacy protection.
If you have any suggestions or problems, you can send it to us by text,screenshot or video record.
Bondee requires the following permissions when you access some of the app's features.
-Albums (storage): storage of photos and videos, and uploading the content in your albums.
-Camera: to take photos, record videos, and scan QR codes.
-Microphone: to record videos and send voice messages.
-Message notifications: to push chat messages and system notifications.
-Contacts: to discover friends who have joined Bondee.\n
Now associating the following conversation history, write a response that appropriately completes the user's latest request while complying with the above instructions.""",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_koala_v1 = Conversation(
    system="BEGINNING OF CONVERSATION:",
    roles=("USER", "GPT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_dolly = Conversation(
    system="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
    roles=("### Instruction", "### Response"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.DOLLY,
    sep="\n\n",
    sep2="### End",
)

conv_oasst = Conversation(
    system="",
    roles=("<|prompter|>", "<|assistant|>"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.OASST_PYTHIA,
    sep="<|endoftext|>",
)

conv_stablelm = Conversation(
    system="""<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
""",
    roles=("<|USER|>", "<|ASSISTANT|>"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.OASST_PYTHIA,
    sep="",
)

conv_templates = {
    "conv_one_shot": conv_one_shot,
    "vicuna_v1.1": conv_vicuna_v1_1,
    "koala_v1": conv_koala_v1,
    "dolly": conv_dolly,
    "oasst": conv_oasst,
    "bondee": conv_bondee,
}


def get_default_conv_template(model_name):
    model_name = model_name.lower()
    if "vicuna" in model_name or "output" in model_name:
        return conv_vicuna_v1_1
    if "bondee" in model_name:
        print("prompt-template: conv_bondee")
        return conv_bondee
    elif "koala" in model_name:
        return conv_koala_v1
    elif "dolly-v2" in model_name:
        return conv_dolly
    elif "oasst" in model_name and "pythia" in model_name:
        return conv_oasst
    elif "stablelm" in model_name:
        return conv_stablelm
    return conv_one_shot


def compute_skip_echo_len(model_name, conv, prompt):
    model_name = model_name.lower()
    if "chatglm" in model_name:
        skip_echo_len = len(conv.messages[-2][1]) + 1
    elif "dolly-v2" in model_name:
        special_toks = ["### Instruction:", "### Response:", "### End"]
        skip_echo_len = len(prompt)
        for tok in special_toks:
            skip_echo_len -= prompt.count(tok) * len(tok)
    elif "oasst" in model_name and "pythia" in model_name:
        special_toks = ["<|prompter|>", "<|assistant|>", "<|endoftext|>"]
        skip_echo_len = len(prompt)
        for tok in special_toks:
            skip_echo_len -= prompt.count(tok) * len(tok)
    elif "stablelm" in model_name:
        special_toks = ["<|SYSTEM|>", "<|USER|>", "<|ASSISTANT|>"]
        skip_echo_len = len(prompt)
        for tok in special_toks:
            skip_echo_len -= prompt.count(tok) * len(tok)
    else:
        skip_echo_len = len(prompt) + 1 - prompt.count("</s>") * 3
    return skip_echo_len


if __name__ == "__main__":
    print(default_conversation.get_prompt())
