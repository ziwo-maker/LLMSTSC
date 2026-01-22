from pyexpat.errors import messages
from utils.my_utils import dump_json, get_state_detail, state2text, getPrompt, action2code, code2action, eight_phase_list, four_phase_list, torch_gc, location_direction_dict

from src.TimeVLM.ts_image_adapter import TimeSeriesImageAdapter
from layers.TimeSeries_To_Image import time_series_to_simple_image
from layers.Learnable_TimeSeries_To_Image import LearnableTimeSeriesToImage
import vllm
import os
import time
import numpy as np
import wandb
from utils.cityflow_env import CityFlowEnv
import utils.config as config
from utils.aft_rank_loss_utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm
import torch
from copy import deepcopy
import re
import json
import shutil
import copy
import random
from collections import deque
from datetime import datetime, timedelta
from types import SimpleNamespace
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result

def _str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y")


class _LocalTSImageModel:
    def __init__(
        self,
        image_size,
        seq_len,
        periodicity,
        norm_const,
        learnable_image,
        three_channel_image,
        device=None,
    ):
        self.config = SimpleNamespace(
            image_size=int(image_size),
            seq_len=int(seq_len),
            periodicity=int(periodicity),
            norm_const=float(norm_const),
            learnable_image=bool(learnable_image),
            save_images=False,
            three_channel_image=bool(three_channel_image),
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.learnable_image_module = None
        if self.config.learnable_image:
            self.learnable_image_module = LearnableTimeSeriesToImage(
                input_dim=3,
                hidden_dim=48,
                output_channels=3 if self.config.three_channel_image else 1,
                image_size=self.config.image_size,
                periodicity=self.config.periodicity,
            ).to(self.device)
            self.learnable_image_module.eval()

    def _infer_device(self):
        if self.learnable_image_module is not None:
            try:
                return next(self.learnable_image_module.parameters()).device
            except StopIteration:
                pass
        return self.device

    def _normalize_input(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev /= self.config.norm_const
        x_enc = x_enc / stdev
        return x_enc, means, stdev

    def vision_augmented_learner(self, x_enc, image_size, context_len, periodicity):
        if self.config.learnable_image:
            images = self.learnable_image_module(x_enc)
        else:
            images = time_series_to_simple_image(x_enc, image_size, context_len, periodicity)
        images = self._normalize_images(images)
        return images

    @staticmethod
    def _normalize_images(images):
        min_vals = images.reshape(images.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_vals = images.reshape(images.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        scale = (max_vals - min_vals).clamp(min=1e-5)
        images = (images - min_vals) / scale
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        return images

def path_check(dic_path):
    if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
        if dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])
    if os.path.exists(dic_path["PATH_TO_MODEL"]):
        if dic_path["PATH_TO_MODEL"] != "model/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_MODEL"])


def copy_conf_file(dic_path, dic_agent_conf, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    json.dump(dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"), indent=4)
    json.dump(dic_traffic_env_conf, open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)


def copy_cityflow_file(dic_path, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["TRAFFIC_FILE"]),
                os.path.join(path, dic_traffic_env_conf["TRAFFIC_FILE"]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["ROADNET_FILE"]),
                os.path.join(path, dic_traffic_env_conf["ROADNET_FILE"]))


class LLM_CGPR_Collector:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.agents = []
        self.env = None
        self.roadnet = roadnet
        self.trafficflow = trafficflow
        self.models = []
        self.generation_kwargs = {}
        self.epoch_num = 0
        self.tokenizer = None
        self.processor = None
        self.llm_model = None
        self.llm_ref_model = None
        self.critic_agents = None
        self.dic_critic_agent_conf = None
        self.training_args = None
        self.trainer_built = False
        self.trainer = None
        self.device = None
        self.fail_log_file = f"./fails/{self.dic_agent_conf['LLM_MODEL']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['ROADNET_FILE']}.json"
        self.fail_logs = []
        self.data_buffer = []
        self.initialize()

    def initialize_llm(self):
        device_map = "auto"

        # init LLM
        llm_path = self.dic_agent_conf["LLM_PATH"]
        llm_path_lower = llm_path.lower()
        if "qwen" in llm_path_lower and "vl" in llm_path_lower:
            # Qwen-VL specific loader (text-only usage; processor used as tokenizer)
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            self.qwen=True
            self.llm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                llm_path,
                dtype="auto",
                device_map=device_map,
            )

            # use processor for text tokenization (no image handling here)
            self.tokenizer = AutoProcessor.from_pretrained(llm_path)
            try:
                self.tokenizer.pad_token_id = 0
            except Exception:
                # processor may not expose pad_token_id; ignore if so
                pass
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )

            # init tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_path,
                padding_side="left"
            )
            self.tokenizer.pad_token_id = 0

        self.generation_kwargs = {
            "min_length": -1,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 1.0,
            "do_sample": False,
            "max_new_tokens": self.dic_agent_conf["NEW_MAX_TOKENS"],
            "num_beam_groups": 4,
            "diversity_penalty": 1.0,
            "num_beams": 4,
            "num_return_sequences": 4
        }

    def initialize_critic(self):
        round_num = 99
        traffic_file = self.dic_traffic_env_conf['TRAFFIC_FILE'].split('.')[0]

        dic_adv_colight_agent_conf_extra = {
            "MODEL_NAME": "AdvancedColight",
            "LIST_STATE_FEATURE": [
                "cur_phase",
                "traffic_movement_pressure_queue_efficient",
                "lane_enter_running_part",
                "adjacency_matrix",
            ],
            "CNN_layers": [[32, 32]],
        }
        dic_critic_agent_conf = merge(dic_adv_colight_agent_conf_extra, config.DIC_BASE_AGENT_CONF)

        if dic_critic_agent_conf["MODEL_NAME"] in self.dic_traffic_env_conf["LIST_MODEL_NEED_TO_UPDATE"]:
            dic_critic_agent_conf["EPSILON"] = 0
            dic_critic_agent_conf["MIN_EPSILON"] = 0

        critic_agents = []
        compare_dic_traffic_env_conf = deepcopy(self.dic_traffic_env_conf)
        compare_dic_traffic_env_conf["LIST_STATE_FEATURE"] = dic_critic_agent_conf["LIST_STATE_FEATURE"]
        for j in range(self.dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = dic_critic_agent_conf["MODEL_NAME"]
            agent = config.DIC_AGENTS[agent_name](
                dic_agent_conf=dic_critic_agent_conf,
                dic_traffic_env_conf=compare_dic_traffic_env_conf,
                dic_path=self.dic_path,
                cnt_round=0,
                intersection_id=str(j)
            )
            critic_agents.append(agent)

        for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
            critic_agents[i].load_network(f'round_{round_num}_inter_0', file_path=f"./model_weights/{dic_critic_agent_conf['MODEL_NAME']}/{traffic_file}/")

        self.critic_agents = critic_agents
        self.dic_critic_agent_conf = dic_critic_agent_conf

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        self.env.reset()
        self.initialize_llm()

    

    def _ensure_ts_history(self, num_intersections):
        if self.ts_history is not None and len(self.ts_history) == num_intersections:
            return
        self.ts_history = [deque(maxlen=self.ts_image_seq_len) for _ in range(num_intersections)]

    def _state_to_features(self, state):
        mode = self.ts_feature_mode
        if mode not in ("movement_total", "movement_queue_cells", "movement_full"):
            mode = "movement_total"
        features = []
        for key in location_direction_dict:
            entry = state.get(key, {})
            queue_len = float(entry.get("queue_len", 0.0))
            cells = list(entry.get("cells", []) or [])
            avg_wait = float(entry.get("avg_wait_time", 0.0))
            if self.ts_cells_len is None:
                self.ts_cells_len = len(cells)
            if self.ts_cells_len:
                if len(cells) < self.ts_cells_len:
                    cells = cells + [0.0] * (self.ts_cells_len - len(cells))
                else:
                    cells = cells[:self.ts_cells_len]
            if mode == "movement_total":
                features.append(queue_len + sum(cells))
            elif mode == "movement_queue_cells":
                features.append(queue_len)
                features.extend(cells)
            else:
                features.append(queue_len)
                features.append(avg_wait)
                features.extend(cells)
        return features

    def _update_ts_history(self, current_states):
        self._ensure_ts_history(len(current_states))
        for i, state in enumerate(current_states):
            features = self._state_to_features(state)
            if self.ts_feature_dim is None:
                self.ts_feature_dim = len(features)
            self.ts_history[i].append(features)

    def _build_ts_batch(self):
        batch = []
        for history in self.ts_history:
            history_list = list(history)
            if not history_list:
                if self.ts_feature_dim is None:
                    return []
                pad = [0.0] * self.ts_feature_dim
                seq = [pad] * self.ts_image_seq_len
            elif len(history_list) < self.ts_image_seq_len:
                pad = history_list[0]
                seq = [pad] * (self.ts_image_seq_len - len(history_list)) + history_list
            else:
                seq = history_list
            batch.append(seq)
        return batch

    def _inject_image_token(self, prompt_text):
        prefix = self.ts_image_prompt_prefix or ""
        token = self.ts_image_token or ""
        if not token and not prefix:
            return prompt_text
        return f"{prefix}{token}\n{prompt_text}"

    def _tensor_images_to_pil(self, images):
        if not torch.is_tensor(images):
            return None
        images = images.detach().cpu()
        if images.ndim != 4:
            return None
        pil_images = []
        for img in images:
            if img.shape[0] == 1:
                array = img.squeeze(0).numpy().astype(np.uint8)
                mode = "L"
            else:
                array = img.permute(1, 2, 0).numpy().astype(np.uint8)
                mode = "RGB"
            pil_images.append(Image.fromarray(array, mode=mode))
        return pil_images

    def _generate_ts_images_local(self, ts_batch):
        if self.ts_image_adapter is None:
            return None
        try:
            images = self.ts_image_adapter.generate(ts_batch)
        except Exception as exc:
            print(f"TS image adapter error: {exc}")
            return None
        return self._tensor_images_to_pil(images)
        self.initialize_critic()

    def collect(self):
        print("================ Start Training ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        # initialize output streams
        done = False
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []
        state = self.env.reset()
        print("end reset")
        current_time = self.env.get_current_time()  # in seconds

        start_time = time.time()
        state_action_log = [[] for _ in range(len(state))]

        # data buffer for training data collection
        # self.llm_model.eval()
        for step_num in tqdm(range(int(total_run_cnt / self.dic_traffic_env_conf['MIN_ACTION_TIME']))):
            if done or current_time >= total_run_cnt:
                break
            action_list = []
            current_states = []

            for i in range(len(state)):
                # log statistic state
                intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
                roads = deepcopy(intersection["roads"])
                statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, self.env)
                state_action_log[i].append({"state": statistic_state, "state_incoming": statistic_state_incoming,
                                            "approaching_speed": mean_speed})
                current_states.append(statistic_state)

            prompts = []
            for s in current_states:
                prompt = getPrompt(state2text(s))
                prompt = prompt[0]['content'] + "\n\n### Instruction:\n" + prompt[1]['content'] + "\n\n### Response:\n"
                prompts.append(prompt)
            inputs = self.tokenizer(prompts, return_tensors="pt", padding="longest")['input_ids'].to('cuda')

            response_ids = self.llm_model.generate(input_ids=inputs, **self.generation_kwargs)
            response_ids = response_ids.reshape(-1, 4, response_ids.size(1))
            responses = []
            for i in range(response_ids.size(0)):
                responses.append(self.tokenizer.batch_decode(response_ids[i], skip_special_tokens=True))

            rewards = []
            all_decoded_responses = []
            all_sampled_rewards = []
            critic_actions = []
            fail_num = 0
            vehicle_nums = self.get_vehicle_num(current_states)
            for i, res in enumerate(responses):
                action_response = responses[i][random.randint(0, 3)][len(prompts[i]):]
                signal_answer_pattern = r'<signal>(.*?)</signal>'
                signals = re.findall(signal_answer_pattern, action_response)
                signal_text = signals[-1] if len(signals) > 0 else "ETWT"
                action_list.append(action2code(signal_text) if signal_text in four_phase_list else 0)
                if len(signals) == 0 or signal_text not in four_phase_list:
                    signal_text = "ETWT"
                    if vehicle_nums[i] != 0:
                        self.fail_logs.append({"state": current_states[i], "response": action_response})
                        dump_json(self.fail_logs, self.fail_log_file)
                        fail_num += 1

                # critic agents
                one_state, _ = self.env.get_state(self.dic_critic_agent_conf["LIST_STATE_FEATURE"])
                critic_agent_action, q_value = self.critic_agents[i].choose_action_with_value(step_num, one_state)
                critic_actions.append(code2action(critic_agent_action[i]))
                rewards.append(q_value[i][action2code(signal_text)])

                # collect responses
                prompt_responses = []
                sampled_rewards = []
                for res_i in range(4):
                    sampled_response = res[res_i][len(prompts[i]):]
                    sampled_signals = re.findall(signal_answer_pattern, sampled_response)
                    sampled_signal_text = sampled_signals[-1] if len(sampled_signals) > 0 else "ETWT"
                    if len(sampled_signals) == 0 or sampled_signal_text not in four_phase_list:
                        sampled_rewards.append(0)
                    else:
                        sampled_rewards.append(float(q_value[i][action2code(sampled_signal_text)]))

                    prompt_responses.append(sampled_response)
                all_decoded_responses.append(prompt_responses)
                all_sampled_rewards.append(sampled_rewards)

            next_state, _, done, _ = self.env.step(action_list)

            for i, res in enumerate(responses):
                if vehicle_nums[i] > 0:
                    new_d = {"query": prompts[i],
                             "responses": all_decoded_responses[i],
                             "scores": all_sampled_rewards[i]}
                    com_score = new_d["scores"][0]
                    all_same = True
                    for s in new_d["scores"]:
                        if s != com_score:
                            all_same = False

                    if not all_same:
                        self.data_buffer.append(new_d)

            # log action
            for i in range(len(state)):
                state_action_log[i][-1]["action"] = eight_phase_list[action_list[i]]

            current_time = self.env.get_current_time()  # in seconds
            state = next_state

            # calculate logger results
            total_reward += sum(rewards)
            print("Rewards:", sum(rewards), "Fail Num:", fail_num)
            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            queue_length_episode.append(sum(queue_length_inter))

            # waiting time
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

            if not os.path.exists("./data/cgpr"):
                os.makedirs("./data/cgpr")
            dump_json(self.data_buffer, f"./data/cgpr/cgpr_{self.dic_traffic_env_conf['TRAFFIC_FILE']}")
            torch_gc()

        # wandb logger
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else self.dic_traffic_env_conf["RUN_COUNTS"]
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

        results = {
            "env/collect_reward": total_reward,
            "env/collect_avg_queue_len": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "env/collect_queuing_vehicle_num": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "env/collect_avg_waiting_time": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "env/collect_avg_travel_time": total_travel_time}
        print("Collect:", results)
        f_state_action = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        dump_json(state_action_log, f_state_action)

        print("Collection time: ", time.time() - start_time)
        self.env.batch_log_2()

        if not os.path.exists("./data/cgpr"):
            os.makedirs("./data/cgpr")
        dump_json(self.data_buffer, f"./data/cgpr/cgpr_{self.dic_traffic_env_conf['TRAFFIC_FILE']}")

    def train_test(self):
        print("================ Start Data Collection ================")
        self.collect()

    def get_vehicle_num(self, states):
        veh_nums = []

        for i in range(len(states)):
            vehicle_num = 0

            for lane in states[i]:
                vehicle_num += states[i][lane]['queue_len']
                for cell in range(len(states[i][lane]['cells'])):
                    vehicle_num += states[i][lane]['cells'][cell]

            veh_nums.append(vehicle_num)

        return veh_nums


class LLM_CGPR_Trainer:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.agents = []
        self.env = None
        self.roadnet = roadnet
        self.trafficflow = trafficflow
        self.models = []
        self.generation_kwargs = {}
        self.epoch_num = 0
        self.tokenizer = None
        self.llm_model = None
        self.llm_ref_model = None
        self.critic_agents = None
        self.dic_critic_agent_conf = None
        self.training_args = None
        self.trainer_built = False
        self.trainer = None
        self.device = None
        self.fail_log_file = f"./fails/{self.dic_agent_conf['LLM_MODEL']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['ROADNET_FILE']}.json"
        self.fail_logs = []
        self.initialize()

    def initialize_llm(self):
        device_map = "auto"

        # init LLM
        llm_path = self.dic_agent_conf["LLM_PATH"]
        llm_path_lower = llm_path.lower()
        if "qwen" in llm_path_lower and "vl" in llm_path_lower:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            self.qwen=True
            # load Qwen-VL (text-only usage)
            self.llm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                llm_path,
                dtype="auto",
                device_map=device_map,
            )

            # processor as tokenizer for text-only
            self.tokenizer = AutoProcessor.from_pretrained(llm_path)
            try:
                self.tokenizer.pad_token_id = 0
            except Exception:
                pass
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=torch.bfloat16,
                # load_in_8bit=True,
                device_map=device_map,
            )
        gradient_accumulation_steps = self.dic_agent_conf["BATCH_SIZE"] // self.dic_agent_conf["MINI_BATCH_SIZE"]
        self.training_args = TrainingArguments(output_dir=f"{self.dic_agent_conf['LLM_OUTPUT_DIR']}_{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}",
                                               num_train_epochs=self.dic_agent_conf["EPOCHS"],
                                               per_device_train_batch_size=self.dic_agent_conf["MINI_BATCH_SIZE"],
                                               per_device_eval_batch_size=self.dic_agent_conf["MINI_BATCH_SIZE"],
                                               gradient_accumulation_steps=gradient_accumulation_steps,
                                               learning_rate=self.dic_agent_conf['LEARNING_RATE'],
                                               bf16=True,
                                               logging_steps=1,
                                               evaluation_strategy="steps",
                                               save_strategy="steps",
                                               eval_steps=50 if 'mix' in self.dic_agent_conf['CGPR_DATA_PATH'] else 10,
                                               save_steps=50 if 'mix' in self.dic_agent_conf['CGPR_DATA_PATH'] else 10,
                                               save_total_limit=3,
                                               load_best_model_at_end=True,
                                               model_max_length=2048)

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            padding_side="left",
            padding=True
        )
        self.tokenizer.pad_token_id = 0

        # init lora
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        # self.llm_model = prepare_model_for_kbit_training(self.llm_model)
        self.llm_model = get_peft_model(self.llm_model, lora_config)

        self.test_generation_kwargs = {
            "min_length": -1,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.1,
            "do_sample": True,
            "max_new_tokens": self.dic_agent_conf["NEW_MAX_TOKENS"],
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        self.env.reset()
        self.initialize_llm()

    def train(self):
        print("================ Start Training ================")
        data = load_dataset("json", data_files=f"./data/cgpr/cgpr_{self.dic_agent_conf['CGPR_DATA_PATH']}")

        train_val = data["train"].train_test_split(test_size=0.1, shuffle=True, seed=2024)
        train_data = train_val["train"].shuffle(seed=2024)
        val_data = train_val["test"].shuffle(seed=2024)

        self.llm_model.train()
        data_module = make_supervised_data_module(self.tokenizer, train_data, val_data, mix=True if 'mix' in self.dic_agent_conf['CGPR_DATA_PATH'] else False)
        self.trainer = RankTrainer(model=self.llm_model, tokenizer=self.tokenizer, args=self.training_args, **data_module)

        # self.llm_model.config.use_cache = False
        # self.llm_model = torch.compile(self.llm_model)
        self.trainer.train()

    def test(self, logger, test_round):
        print("================ Start Test ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        # initialize output streams
        done = False
        state = self.env.reset()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []
        print("end reset")
        current_time = self.env.get_current_time()  # in seconds

        start_time = time.time()
        state_action_log = [[] for _ in range(len(state))]
        self._load_traffic_history()

        # self.llm_model.eval()
        for step_num in tqdm(range(int(total_run_cnt / self.dic_traffic_env_conf['MIN_ACTION_TIME']))):
            if done or current_time >= total_run_cnt:
                break
            action_list = []
            current_states = []

            for i in range(len(state)):
                # log statistic state
                intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
                roads = deepcopy(intersection["roads"])
                statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, self.env)
                state_action_log[i].append({"state": statistic_state, "state_incoming": statistic_state_incoming,
                                            "approaching_speed": mean_speed})
                current_states.append(statistic_state)

            prompts = []
            for s in current_states:
                prompt = getPrompt(state2text(s))
                prompt = prompt[0]['content'] + "\n\n### Instruction:\n" + prompt[1]['content'] + "\n\n### Response:\n"
                prompts.append(prompt)
            inputs = self.tokenizer(prompts, truncation=True, max_length=2048, padding=True, return_tensors='pt').to('cuda')

            # response_ids = self.llm_model.generate(input_ids=inputs["input_ids"], **self.test_generation_kwargs)
            # 构造全 0 的 token_type_ids (假设这是纯文本部分，或者模型默认处理)
            # 注意：如果是包含图片的输入，这里不能简单的全 0，必须沿用预处理时的逻辑
            current_token_type_ids = torch.zeros_like(current_input_ids)

            response_ids = self.llm_model.generate(
                input_ids=current_input_ids,
                token_type_ids=current_token_type_ids, 
                **self.test_generation_kwargs
            )
            responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            fail_num = 0
            vehicle_nums = self.get_vehicle_num(current_states)
            for i, res in enumerate(responses):
                res = res[len(prompts[i]):]
                signal_answer_pattern = r'<signal>(.*?)</signal>'
                signals = re.findall(signal_answer_pattern, res)
                signal_text = signals[-1] if len(signals) > 0 else "ETWT"
                action_list.append(action2code(signal_text) if signal_text in four_phase_list else 0)
                if len(signals) == 0 or signal_text not in four_phase_list:
                    signal_text = "ETWT"
                    if vehicle_nums[i] != 0:
                        self.fail_logs.append({"state": current_states[i], "response": res})
                        dump_json(self.fail_logs, self.fail_log_file)
                        fail_num += 1

            next_state, rewards, done, _ = self.env.step(action_list)

            # log action
            for i in range(len(state)):
                state_action_log[i][-1]["action"] = eight_phase_list[action_list[i]]

            current_time = self.env.get_current_time()  # in seconds
            state = next_state

            # calculate logger results
            total_reward += sum(rewards)
            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            queue_length_episode.append(sum(queue_length_inter))
            print("Fail Num:", fail_num, "Queuing Vehicles:", sum(queue_length_episode))

            # waiting time
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

        # wandb logger
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else self.dic_traffic_env_conf["RUN_COUNTS"]
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

        results = {
            "env/test_reward": total_reward,
            "env/test_avg_queue_len": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "env/test_queuing_vehicle_num": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "env/test_avg_waiting_time": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "env/test_avg_travel_time": total_travel_time}
        logger.log(results)
        print("Test Round:", test_round, results)
        f_state_action = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        dump_json(state_action_log, f_state_action)
        print("Testing time: ", time.time() - start_time)

        self.env.batch_log_2()

        return results

    def train_test(self):
        print("================ Start PPO Fine-Tuning ================")
        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)
        logger = wandb.init(
            project=self.dic_traffic_env_conf['PROJECT_NAME'],
            group=f"{self.dic_traffic_env_conf['MODEL_NAME']}-{self.roadnet}-{self.trafficflow}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases",
            name=f"{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}",
            config=all_config,
        )

        rounds = self.dic_traffic_env_conf["NUM_ROUNDS"]
        last_10_results = {"env/test_reward_over": [],
                           "env/test_avg_queue_len_over": [],
                           "env/test_queuing_vehicle_num_over": [],
                           "env/test_avg_waiting_time_over": [],
                           "env/test_avg_travel_time_over": []}
        for r in range(rounds):
            # train
            self.train()

            # test
            results = self.test(logger, r)
            for ele in last_10_results:
                last_10_results[ele].append(results[ele[:-5]])

            main_path = f"{self.dic_agent_conf['LLM_OUTPUT_DIR']}_{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}"
            ckpt_path = f"{self.dic_agent_conf['LLM_OUTPUT_DIR']}_{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}/my_checkpoint_{r}"
            if not os.path.isdir(main_path):
                os.mkdir(main_path)
            if not os.path.isdir(ckpt_path):
                os.mkdir(ckpt_path)

            self.llm_model.save_pretrained(
                f"{self.dic_agent_conf['LLM_OUTPUT_DIR']}_{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}/my_checkpoint_{r}")

        logger.log(last_10_results)
        wandb.finish()

        self.llm_model.save_pretrained(f"{self.dic_agent_conf['LLM_OUTPUT_DIR']}_{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}")

    '''
    ======================= Class Utils =======================
    '''
    def get_vehicle_num(self, states):
        veh_nums = []

        for i in range(len(states)):
            vehicle_num = 0

            for lane in states[i]:
                vehicle_num += states[i][lane]['queue_len']
                for cell in range(len(states[i][lane]['cells'])):
                    vehicle_num += states[i][lane]['cells'][cell]

            veh_nums.append(vehicle_num)

        return veh_nums


class LLM_Inference:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow, ts_image_adapter=None):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.agents = []
        self.env = None
        self.roadnet = roadnet
        self.trafficflow = trafficflow
        self.models = []
        self.generation_kwargs = {}
        self.epoch_num = 0
        self.tokenizer = None
        self.llm_model = None
        self.llm_ref_model = None
        self.dic_critic_agent_conf = None
        self.training_args = None
        self.trainer_built = False
        self.trainer = None
        self.device = None

        self.ts_image_adapter = ts_image_adapter
        self.ts_history = None
        self.ts_feature_dim = None
        self.ts_cells_len = None
        self.ts_image_service_url = str(self.dic_agent_conf.get("TS_IMAGE_SERVICE_URL", "")).strip()
        try:
            self.ts_image_seq_len = int(self.dic_agent_conf.get("TS_IMAGE_SEQ_LEN", 0) or 0)
        except (TypeError, ValueError):
            self.ts_image_seq_len = 0
        if self.ts_image_adapter is not None and self.ts_image_seq_len <= 0:
            adapter_seq_len = getattr(self.ts_image_adapter, "seq_len", None)
            if adapter_seq_len is not None:
                self.ts_image_seq_len = int(adapter_seq_len)
        self.ts_image_token = str(self.dic_agent_conf.get("TS_IMAGE_TOKEN", "<image>"))
        self.ts_image_prompt_prefix = str(self.dic_agent_conf.get("TS_IMAGE_PROMPT_PREFIX", ""))
        self.ts_feature_mode = str(self.dic_agent_conf.get("TS_FEATURE_MODE", "movement_total")).strip().lower()
        try:
            self.ts_image_timeout = float(self.dic_agent_conf.get("TS_IMAGE_TIMEOUT", 5.0))
        except (TypeError, ValueError):
            self.ts_image_timeout = 5.0
        try:
            self.ts_image_size = int(self.dic_agent_conf.get("TS_IMAGE_SIZE", 56) or 56)
        except (TypeError, ValueError):
            self.ts_image_size = 56 
        try:
            self.ts_image_periodicity = int(self.dic_agent_conf.get("TS_IMAGE_PERIODICITY", 24) or 24)
        except (TypeError, ValueError):
            self.ts_image_periodicity = 24
        try:
            self.ts_image_norm_const = float(self.dic_agent_conf.get("TS_IMAGE_NORM_CONST", 0.4) or 0.4)
        except (TypeError, ValueError):
            return
        self.ts_image_norm_const = 0.4
        self.ts_image_learnable = _str2bool(self.dic_agent_conf.get("TS_IMAGE_LEARNABLE", False))
        self.ts_image_three_channel = _str2bool(self.dic_agent_conf.get("TS_IMAGE_THREE_CHANNEL", True))

        self._traffic_history_interval = None
        self._traffic_history_intersection_ids = None
        self._traffic_history_intersection_index = None
        self._traffic_history_movement_keys = None
        self._traffic_history_output_path = None
        self._traffic_history_last_written_bin = -1
        self._traffic_history_base_dt = None
     
        if not os.path.exists("./fails"):
            os.mkdir("./fails")
        self.fail_log_file = f"./fails/{self.dic_agent_conf['LLM_MODEL']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['ROADNET_FILE']}.json"
        self.fail_logs = []
        self._init_ts_image_client()
        if self.ts_image_adapter is None:
            self._init_ts_image_adapter()
        self.initialize()

    def initialize_llm(self):
        device_map = "auto"

        # init LLM
        llm_path = self.dic_agent_conf["LLM_PATH"]
        llm_path_lower = llm_path.lower()
        if "qwen" in llm_path_lower and "vl" in llm_path_lower:
            self.qwen=True
            self.llm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                llm_path,
                dtype="auto",
                device_map=device_map,
            )

            # processor as tokenizer for text-only usage
            self.processor = AutoProcessor.from_pretrained(llm_path, padding_side="left")
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_path,
                padding_side="left",
                padding=True
            )
            try:
                self.tokenizer.pad_token_id = 0
            except Exception:
                pass
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True
            )

            # init tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_path,
                padding_side="left",
                padding=True
            )
            self.tokenizer.pad_token_id = 0

            self.processor = AutoProcessor.from_pretrained(llm_path, padding_side="left")

        self.test_generation_kwargs = {
            "min_length": -1,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.1,
            "do_sample": True,
            "max_new_tokens": self.dic_agent_conf["NEW_MAX_TOKENS"],
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None)
        }

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        self.env.reset()
        self.initialize_llm()

    def _init_ts_image_client(self):
        if not self.ts_image_service_url or self.ts_image_seq_len <= 0:
            return


    def _init_ts_image_adapter(self):
        if self.ts_image_seq_len <= 0:
            return
        try:
            model = _LocalTSImageModel(
                image_size=self.ts_image_size,
                seq_len=self.ts_image_seq_len,
                periodicity=self.ts_image_periodicity,
                norm_const=self.ts_image_norm_const,
                learnable_image=self.ts_image_learnable,
                three_channel_image=self.ts_image_three_channel,
            )
            self.ts_image_adapter = TimeSeriesImageAdapter(model)
        except Exception as exc:
            print(f"TS image adapter init error: {exc}")
            self.ts_image_adapter = None

    def _ensure_ts_history(self, num_intersections):
        if self.ts_history is not None and len(self.ts_history) == num_intersections:
            return
        self.ts_history = [deque(maxlen=self.ts_image_seq_len) for _ in range(num_intersections)]

    def _state_to_features(self, state):
        mode = self.ts_feature_mode
        if mode not in ("movement_total", "movement_queue_cells", "movement_full"):
            mode = "movement_total"
        features = []
        for key in location_direction_dict:
            entry = state.get(key, {})
            queue_len = float(entry.get("queue_len", 0.0))
            cells = list(entry.get("cells", []) or [])
            avg_wait = float(entry.get("avg_wait_time", 0.0))
            if self.ts_cells_len is None:
                self.ts_cells_len = len(cells)
            if self.ts_cells_len:
                if len(cells) < self.ts_cells_len:
                    cells = cells + [0.0] * (self.ts_cells_len - len(cells))
                else:
                    cells = cells[:self.ts_cells_len]
            if mode == "movement_total":
                features.append(queue_len + sum(cells))
            elif mode == "movement_queue_cells":
                features.append(queue_len)
                features.extend(cells)
            else:
                features.append(queue_len)
                features.append(avg_wait)
                features.extend(cells)
        return features

    def _update_ts_history(self, current_states):
        self._ensure_ts_history(len(current_states))
        for i, state in enumerate(current_states):
            features = self._state_to_features(state)
            if self.ts_feature_dim is None:
                self.ts_feature_dim = len(features)
            self.ts_history[i].append(features)

    def _build_ts_batch(self):
        batch = []
        for history in self.ts_history:
            history_list = list(history)
            if not history_list:
                if self.ts_feature_dim is None:
                    return []
                pad = [0.0] * self.ts_feature_dim
                seq = [pad] * self.ts_image_seq_len
            elif len(history_list) < self.ts_image_seq_len:
                pad = history_list[0]
                seq = [pad] * (self.ts_image_seq_len - len(history_list)) + history_list
            else:
                seq = history_list
            batch.append(seq)
        return batch

    def _traffic_count_suffix(self, interval_seconds):
        if interval_seconds == 60:
            return "minute"
        if interval_seconds == 3600:
            return "hour"
        return f"{interval_seconds}s"

    def _resolve_traffic_history_output_path(self, interval_seconds):
        path = str(self.dic_traffic_env_conf.get("TRAFFIC_COUNT_HISTORY_PATH", "")).strip()
        if path:
            return path
        output_dir = self.dic_traffic_env_conf.get(
            "TRAFFIC_COUNT_OUTPUT_DIR", self.dic_path["PATH_TO_WORK_DIRECTORY"]
        )
        base_name = self.dic_traffic_env_conf.get("TRAFFIC_COUNT_BASENAME", "traffic_counts")
        suffix = self._traffic_count_suffix(interval_seconds)
        return os.path.join(output_dir, f"{base_name}_{suffix}.jsonl")

    def _load_traffic_history(self):
        if self._traffic_history_interval is not None:
            return
        if self.env is None:
            return
        interval = int(self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 1) or 1)
        current_time = self.env.get_current_time()
        needs_reinit = (
            not getattr(self.env, "traffic_count_enabled", False)
            or getattr(self.env, "traffic_count_mode", "") != "intersection_movement"
            or interval not in getattr(self.env, "traffic_count_intervals", [])
        )
        if needs_reinit and current_time in (None, 0, 0.0):
            self.dic_traffic_env_conf["ENABLE_TRAFFIC_COUNT"] = True
            self.dic_traffic_env_conf["TRAFFIC_COUNT_MODE"] = "intersection_movement"
            self.dic_traffic_env_conf["TRAFFIC_COUNT_OUTPUT_FORMAT"] = "jsonl"
            intervals = self.dic_traffic_env_conf.get("TRAFFIC_COUNT_INTERVALS", None)
            if intervals is None:
                intervals = [interval]
            elif isinstance(intervals, int):
                intervals = [intervals]
            intervals = [int(x) for x in intervals if int(x) > 0]
            if interval not in intervals:
                intervals.append(interval)
            self.dic_traffic_env_conf["TRAFFIC_COUNT_INTERVALS"] = sorted(set(intervals))
            try:
                self.env._init_traffic_counter()
            except Exception as exc:
                print(f"Traffic counter init failed: {exc}")
        elif needs_reinit:
            print("Traffic counter not configured for MIN_ACTION_TIME; history may be empty.")

        if not getattr(self.env, "traffic_count_enabled", False):
            return
        if getattr(self.env, "traffic_count_mode", "") != "intersection_movement":
            return

        intervals = getattr(self.env, "traffic_count_intervals", [])
        if interval in intervals:
            self._traffic_history_interval = interval
        elif intervals:
            self._traffic_history_interval = min(intervals)
            print(
                f"Using traffic interval {self._traffic_history_interval} "
                f"(MIN_ACTION_TIME={interval} not available)."
            )
        else:
            self._traffic_history_interval = interval

        self._traffic_history_intersection_ids = list(getattr(self.env, "traffic_intersection_ids", []) or [])
        self._traffic_history_intersection_index = {
            inter_id: idx for idx, inter_id in enumerate(self._traffic_history_intersection_ids)
        }
        self._traffic_history_movement_keys = list(
            getattr(self.env, "traffic_movement_keys", []) or ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
        )
        self._traffic_history_last_written_bin = -1
        self._traffic_history_base_dt = getattr(self.env, "traffic_base_dt", None) or datetime(1970, 1, 1)
        output_path = self._resolve_traffic_history_output_path(self._traffic_history_interval)
        self._traffic_history_output_path = output_path
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8"):
                pass

    def _get_traffic_history_until(self, intersection_id, current_time_s):
        self._load_traffic_history()
        interval = self._traffic_history_interval
        if interval is None:
            return []
        counts = self.env.traffic_counts_by_interval.get(interval)
        if counts is None or counts.size == 0:
            return []
        inter_idx = self._traffic_history_intersection_index.get(intersection_id)
        if inter_idx is None:
            return []
        num_mv = len(self._traffic_history_movement_keys)
        current_bin = int(current_time_s) // interval
        if current_bin < 0:
            return []
        if current_bin >= counts.shape[0]:
            current_bin = counts.shape[0] - 1
        offset = inter_idx * num_mv
        slice_rows = counts[: current_bin + 1, offset : offset + num_mv]
        history = []
        for row in slice_rows:
            movement_counts = {
                key: int(row[idx]) for idx, key in enumerate(self._traffic_history_movement_keys)
            }
            history.append(movement_counts)
        return history

    def _append_traffic_history_jsonl(self, current_time_s):
        self._load_traffic_history()
        interval = self._traffic_history_interval
        if interval is None or not self._traffic_history_output_path:
            return
        counts = self.env.traffic_counts_by_interval.get(interval)
        if counts is None or counts.size == 0:
            return
        current_bin = int(current_time_s) // interval
        if current_bin < 0:
            return
        if current_bin >= counts.shape[0]:
            current_bin = counts.shape[0] - 1
        if current_bin <= self._traffic_history_last_written_bin:
            return
        base_dt = self._traffic_history_base_dt or datetime(1970, 1, 1)
        num_mv = len(self._traffic_history_movement_keys)
        with open(self._traffic_history_output_path, "a", encoding="utf-8") as handle:
            for bin_idx in range(self._traffic_history_last_written_bin + 1, current_bin + 1):
                date_str = (base_dt + timedelta(seconds=bin_idx * interval)).strftime("%Y-%m-%d %H:%M:%S")
                row = counts[bin_idx, :]
                for inter_idx, inter_id in enumerate(self._traffic_history_intersection_ids or []):
                    offset = inter_idx * num_mv
                    movement_counts = {
                        key: int(row[offset + mv_i]) for mv_i, key in enumerate(self._traffic_history_movement_keys)
                    }
                    record = {
                        "date": date_str,
                        "interval_s": int(interval),
                        "intersection_id": inter_id,
                        "movement_counts": movement_counts,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._traffic_history_last_written_bin = current_bin

    def _inject_image_token(self, prompt_text):
        prefix = self.ts_image_prompt_prefix or ""
        token = self.ts_image_token or ""
        if not token and not prefix:
            return prompt_text
        return f"{prefix}{token}\n{prompt_text}"

    def _tensor_images_to_pil(self, images):
        if not torch.is_tensor(images):
            return None
        images = images.detach().cpu()
        if images.ndim != 4:
            return None
        pil_images = []
        for img in images:
            if img.shape[0] == 1:
                array = img.squeeze(0).numpy().astype(np.uint8)
                mode = "L"
            else:
                array = img.permute(1, 2, 0).numpy().astype(np.uint8)
                mode = "RGB"
            pil_images.append(Image.fromarray(array, mode=mode))
        return pil_images

    def _generate_ts_images_local(self, ts_batch):
        if self.ts_image_adapter is None:
            return None
        try:
            images = self.ts_image_adapter.generate(ts_batch)
        except Exception as exc:
            print(f"TS image adapter error: {exc}")
            return None
        return self._tensor_images_to_pil(images)

    def test(self, logger, test_round):
        print("================ Start Test ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]

        # initialize output streams
        done = False
        state = self.env.reset()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []

        print("end reset")
        current_time = self.env.get_current_time()  # in seconds
        start_time = time.time()

        state_action_log = [[] for _ in range(len(state))]
        self._load_traffic_history()

        # self.llm_model.eval()
        for step_num in tqdm(range(int(total_run_cnt / self.dic_traffic_env_conf["MIN_ACTION_TIME"]))):
            if done or current_time >= total_run_cnt:
                break

            action_list = []
            current_states = []

            for i in range(len(state)):
                # log statistic state
                intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
                roads = deepcopy(intersection["roads"])
                statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, self.env)

                state_action_log[i].append(
                    {
                        "state": statistic_state,
                        "state_incoming": statistic_state_incoming,
                        "approaching_speed": mean_speed,
                    }
                )
                current_states.append(statistic_state)

            prompts = []
            ts_images = []

            self._append_traffic_history_jsonl(current_time)

            movement_keys = self._traffic_history_movement_keys or ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
            ts_adapter = self.ts_image_adapter
            if ts_adapter is None:
                raise RuntimeError(
                    "ts_image_adapter is not initialized; create it in run_open_LLM.py and pass it to LLM_Inference."
                )

            inferred_dim = None
            for idx, s in enumerate(current_states):
                state_action_log[idx].append({"state": s})
                intersection_id = self.env.list_intersection[idx].inter_name
                traffic_history = self._get_traffic_history_until(intersection_id, current_time)

                if not traffic_history:
                    raise ValueError(f"No traffic history for intersection {intersection_id} at t={current_time}.")

                ts_seq = [[float(step.get(key, 0.0)) for key in movement_keys] for step in traffic_history]

                try:
                    current_dim = len(ts_seq[0])
                except (TypeError, IndexError):
                    current_dim = self.ts_feature_dim or 3

                if inferred_dim is None:
                    inferred_dim = current_dim
                    if self.ts_feature_dim is None:
                        self.ts_feature_dim = inferred_dim

                    if getattr(ts_adapter, "learnable_image", False):
                        adapter_dim = getattr(ts_adapter, "input_dim", None)
                        if adapter_dim is not None and adapter_dim != inferred_dim:
                            raise ValueError(
                                "TS image adapter input_dim mismatch; reinitialize in main with input_dim="
                                f"{inferred_dim}."
                            )

                elif current_dim != inferred_dim:
                    raise ValueError(
                        f"Traffic history feature dim mismatch for {intersection_id}: "
                        f"expected {inferred_dim}, got {current_dim}."
                    )

                ts_images_local = ts_adapter.generate_images(ts_seq)
                if not ts_images_local:
                    raise RuntimeError(f"TS image adapter returned no images for intersection {intersection_id}.")

                ts_image = ts_images_local[0]
                ts_images.append(ts_image)

                prompt = getPrompt(state2text(s))
                prompt = prompt[0]["content"] + "\n\n### Instruction:\n" + prompt[1]["content"] + "\n\n### Response:\n"



                prompts.append(prompt)

            if hasattr(self, "qwen") and self.qwen is not None and self.qwen is True:
                inputs_text = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": f"<start_of_image> {p}"}],  # 这里 content 直接用字符串 p
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for p in prompts
                ]
                # 第二步：调用 tokenizer/processor 时，显式指定 text 参数
                # 关键修改：加上 text=... 并传入 ts_images 作为多模态输入
            else:
                inputs_text = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": f"<start_of_image> {p}"}],  # 这里 content 直接用字符串 p
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for p in prompts
                ]
            

            images_batched = [[img] for img in ts_images]

            inputs = self.processor(text=inputs_text, images=images_batched, padding=True, return_tensors="pt")
            responses = []
            previous_flag = 0
            for i in range(len(current_states)):
                if (i + 1) % 16 == 0 or i + 1 >= len(current_states):
                    current_inputs = {k: v[previous_flag : i + 1] for k, v in inputs.items()}
                    
                    response_ids = self.llm_model.generate(**current_inputs, **self.test_generation_kwargs)
                    cur_response = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

                    responses += cur_response
                    previous_flag = i + 1

            fail_num = 0
            vehicle_nums = self.get_vehicle_num(current_states)
            critic_actions = []

            for i, res in enumerate(responses):
                res = res[len(prompts[i]) :]
                signal_answer_pattern = r"<signal>(.*?)</signal>"
                signals = re.findall(signal_answer_pattern, res)
                signal_text = signals[-1] if len(signals) > 0 else "ETWT"

                action_list.append(action2code(signal_text) if signal_text in four_phase_list else 0)

                if len(signals) == 0 or signal_text not in four_phase_list:
                    signal_text = "ETWT"
                    if vehicle_nums[i] != 0:
                        self.fail_logs.append({"state": current_states[i], "response": res})
                        dump_json(self.fail_logs, self.fail_log_file)
                        fail_num += 1

                state_action_log[i][-1]["response"] = res
                state_action_log[i][-1]["action"] = eight_phase_list[action_list[i]]

            next_state, _, done, _ = self.env.step(action_list)
            rewards = self.get_norm_reward(next_state)  # my reward

            current_time = self.env.get_current_time()  # in seconds
            state = next_state

            # calculate logger results
            total_reward += sum(rewards)

            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature["lane_num_waiting_vehicle_in"]))
            queue_length_episode.append(sum(queue_length_inter))

            print("Fail Num:", fail_num, "Queuing Vehicles:", sum(queue_length_episode))

            # waiting time
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]["time"])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

            # wandb logger
            vehicle_travel_times = {}
            for inter in self.env.list_intersection:
                arrive_left_times = inter.dic_vehicle_arrive_leave_time
                for veh in arrive_left_times:
                    if "shadow" in veh:
                        continue

                    enter_time = arrive_left_times[veh]["enter_time"]
                    leave_time = arrive_left_times[veh]["leave_time"]

                    if not np.isnan(enter_time):
                        leave_time = (
                            leave_time
                            if not np.isnan(leave_time)
                            else self.dic_traffic_env_conf["RUN_COUNTS"]
                        )
                        if veh not in vehicle_travel_times:
                            vehicle_travel_times[veh] = [leave_time - enter_time]
                        else:
                            vehicle_travel_times[veh].append(leave_time - enter_time)

            total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

            results = {
                "test_reward_over": total_reward,
                "test_avg_queue_len_over": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
                "test_queuing_vehicle_num_over": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
                "test_avg_waiting_time_over": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
                "test_avg_travel_time_over": total_travel_time,
            }
            logger.log(results)
            print("Test Round:", test_round, results)

        f_state_action = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        dump_json(state_action_log, f_state_action)

        print("Testing time: ", time.time() - start_time)
        self.env.batch_log_2()
        return results


    def train_test(self):
        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)
        logger = wandb.init(
            project=self.dic_traffic_env_conf['PROJECT_NAME'],
            group=f"{self.dic_traffic_env_conf['MODEL_NAME']}-{self.roadnet}-{self.trafficflow}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases",
            name=f"{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}",
            config=all_config,
        )

        self.test(logger, 0)
        wandb.finish()

    '''
    ======================= Class Utils =======================
    '''
    def get_vehicle_num(self, states):
        veh_nums = []

        for i in range(len(states)):
            vehicle_num = 0

            for lane in states[i]:
                vehicle_num += states[i][lane]['queue_len']
                for cell in range(len(states[i][lane]['cells'])):
                    vehicle_num += states[i][lane]['cells'][cell]

            veh_nums.append(vehicle_num)

        return veh_nums

    def get_norm_reward(self, state):
        rewards = []

        for i in range(len(state)):
            vehicle_num = 0
            queue_length = 0

            intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
            roads = deepcopy(intersection["roads"])
            statistic_state, _, _ = get_state_detail(roads, self.env)
            for lane in statistic_state:
                queue_length += statistic_state[lane]['queue_len']

                vehicle_num += statistic_state[lane]['queue_len']
                for cell in range(len(statistic_state[lane]['cells'])):
                    vehicle_num += statistic_state[lane]['cells'][cell]

            reward = -(queue_length / vehicle_num) if vehicle_num > 0.0 else -0.0
            rewards.append(reward)

        return rewards


class LLM_Inference_VLLM:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.agents = []
        self.env = None
        self.roadnet = roadnet
        self.trafficflow = trafficflow
        self.models = []
        self.generation_kwargs = {}
        self.epoch_num = 0
        self.tokenizer = None
        self.llm_model = None
        self.llm_ref_model = None
        self.dic_critic_agent_conf = None
        self.training_args = None
        self.trainer_built = False
        self.trainer = None
        self.device = None
        self.ts_image_client = None
        self.ts_image_adapter = None
        self.ts_history = None
        self.ts_feature_dim = None
        self.ts_cells_len = None
        self.ts_image_service_url = str(self.dic_agent_conf.get("TS_IMAGE_SERVICE_URL", "")).strip()
        try:
            self.ts_image_seq_len = int(self.dic_agent_conf.get("TS_IMAGE_SEQ_LEN", 0) or 0)
        except (TypeError, ValueError):
            self.ts_image_seq_len = 0
        self.ts_image_token = str(self.dic_agent_conf.get("TS_IMAGE_TOKEN", "<image>"))
        self.ts_image_prompt_prefix = str(self.dic_agent_conf.get("TS_IMAGE_PROMPT_PREFIX", ""))
        self.ts_feature_mode = str(self.dic_agent_conf.get("TS_FEATURE_MODE", "movement_total")).strip().lower()
        try:
            self.ts_image_timeout = float(self.dic_agent_conf.get("TS_IMAGE_TIMEOUT", 5.0))
        except (TypeError, ValueError):
            self.ts_image_timeout = 5.0
        try:
            self.ts_image_size = int(self.dic_agent_conf.get("TS_IMAGE_SIZE", 56) or 56)
        except (TypeError, ValueError):
            self.ts_image_size = 56
        try:
            self.ts_image_periodicity = int(self.dic_agent_conf.get("TS_IMAGE_PERIODICITY", 24) or 24)
        except (TypeError, ValueError):
            self.ts_image_periodicity = 24
        try:
            self.ts_image_norm_const = float(self.dic_agent_conf.get("TS_IMAGE_NORM_CONST", 0.4) or 0.4)
        except (TypeError, ValueError):
            self.ts_image_norm_const = 0.4
        self.ts_image_learnable = _str2bool(self.dic_agent_conf.get("TS_IMAGE_LEARNABLE", False))
        self.ts_image_three_channel = _str2bool(self.dic_agent_conf.get("TS_IMAGE_THREE_CHANNEL", True))
     
        if not os.path.exists("./fails"):
            os.mkdir("./fails")
        self.fail_log_file = f"./fails/{self.dic_agent_conf['LLM_MODEL']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['ROADNET_FILE']}.json"
        self.fail_logs = []
        self._init_ts_image_client()
        self._init_ts_image_adapter()
        self.initialize()

    def _infer_tensor_parallel_size(self):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            ids = [x.strip() for x in visible.split(",") if x.strip() != ""]
            if len(ids) == 1 and "-" in ids[0]:
                parts = ids[0].split("-")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    start = int(parts[0])
                    end = int(parts[1])
                    if end >= start:
                        return end - start + 1
            if ids:
                return len(ids)

        count = torch.cuda.device_count()
        return count if count > 0 else 1
    def _init_ts_image_client(self):
        if not self.ts_image_service_url or self.ts_image_seq_len <= 0:
            return


    def _init_ts_image_adapter(self):
        if self.ts_image_seq_len <= 0:
            return
        try:
            model = _LocalTSImageModel(
                image_size=self.ts_image_size,
                seq_len=self.ts_image_seq_len,
                periodicity=self.ts_image_periodicity,
                norm_const=self.ts_image_norm_const,
                learnable_image=self.ts_image_learnable,
                three_channel_image=self.ts_image_three_channel,
            )
            self.ts_image_adapter = TimeSeriesImageAdapter(model)
        except Exception as exc:
            print(f"TS image adapter init error: {exc}")
            self.ts_image_adapter = None
    def initialize_llm(self):
        device_map = "auto"

        # init LLM
        llm_path = self.dic_agent_conf["LLM_PATH"]
        llm_path_lower = llm_path.lower()
        if "qwen" in llm_path_lower and "vl" in llm_path_lower:
            # vllm may not support Qwen-VL; fallback to transformers loader
            
            self.qwen=True
            print("Warning: detected Qwen-VL model in LLM_Inference_VLLM.initialize_llm — falling back to transformers loader")
            self.llm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                llm_path,
                dtype="auto",
                device_map=device_map,
            )

            # processor as tokenizer for text-only usage
            self.tokenizer = AutoProcessor.from_pretrained(llm_path)
            try:
                self.tokenizer.pad_token_id = 0
            except Exception:
                pass

            # set generation kwargs for transformer usage (note: calling code expects vllm usage)
            self.generation_kwargs = {
                "top_k": 50,
                "top_p": 1.0,
                "temperature": 0.1,
                "max_new_tokens": self.dic_agent_conf["NEW_MAX_TOKENS"]
            }
        else:
            tensor_parallel_size = self.dic_agent_conf.get("TENSOR_PARALLEL_SIZE", 0)
            try:
                tensor_parallel_size = int(tensor_parallel_size)
            except (TypeError, ValueError):
                tensor_parallel_size = 0
            if tensor_parallel_size < 1:
                tensor_parallel_size = self._infer_tensor_parallel_size()

            available_gpus = torch.cuda.device_count()
            if available_gpus > 0 and tensor_parallel_size > available_gpus:
                print(f"Warning: TENSOR_PARALLEL_SIZE={tensor_parallel_size} > visible GPUs={available_gpus}; "
                      f"clamping to {available_gpus}.")
                tensor_parallel_size = available_gpus

            self.llm_model = vllm.LLM(
                model=llm_path,
                tokenizer=llm_path,
                dtype=torch.bfloat16,
                tensor_parallel_size=max(tensor_parallel_size, 1)
            )

            # init tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_path,
                padding_side="left",
                padding=True
            )
            self.tokenizer.pad_token_id = 0

            test_generation_kwargs = {
                "top_k": 50,
                "top_p": 1.0,
                "temperature": 0.1,
                "max_tokens": 2048 + self.dic_agent_conf["NEW_MAX_TOKENS"]
            }
            self.generation_kwargs = vllm.SamplingParams(**test_generation_kwargs)

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        self.env.reset()
        self.initialize_llm()

    def test(self, logger, test_round):
        print("================ Start Test ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        # initialize output streams
        done = False
        state = self.env.reset()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []
        print("end reset")
        current_time = self.env.get_current_time()  # in seconds

        start_time = time.time()
        state_action_log = [[] for _ in range(len(state))]

        # self.llm_model.eval()
        for step_num in tqdm(range(int(total_run_cnt / self.dic_traffic_env_conf['MIN_ACTION_TIME']))):
            if done or current_time >= total_run_cnt:
                break
            action_list = []
            current_states = []

 
            ts_images=[]
            prompts = []
            prompt_items = []
            for idx, s in enumerate(current_states):
                prompt = getPrompt(state2text(s))
                prompt = prompt[0]['content'] + "\n\n### Instruction:\n" + prompt[1]['content'] + "\n\n### Response:\n"
                if ts_images is not None:
                    prompt = self._inject_image_token(prompt)
                prompts.append(prompt)
                if ts_images is not None:
                    prompt_items.append({"prompt": prompt, "multi_modal_data": {"image": ts_images[idx]}})
                else:
                    prompt_items.append(prompt)

            responses = []
            previous_flag = 0
            for i in range(len(current_states)):
                if (i + 1) % 16 == 0 or i + 1 >= len(current_states):
                    responses_meta = self.llm_model.generate(
                        prompts=prompt_items[previous_flag:i + 1],
                        sampling_params=self.generation_kwargs
                    )
                    responses += [res.outputs[0].text for res in responses_meta]
                    previous_flag = i + 1

            fail_num = 0
            vehicle_nums = self.get_vehicle_num(current_states)
            for i, res in enumerate(responses):
                res = res[len(prompts[i]):]
                signal_answer_pattern = r'<signal>(.*?)</signal>'
                signals = re.findall(signal_answer_pattern, res)
                signal_text = signals[-1] if len(signals) > 0 else "ETWT"
                action_list.append(action2code(signal_text) if signal_text in four_phase_list else 0)
                if len(signals) == 0 or signal_text not in four_phase_list:
                    signal_text = "ETWT"
                    if vehicle_nums[i] != 0:
                        self.fail_logs.append({"state": current_states[i], "response": res})
                        dump_json(self.fail_logs, self.fail_log_file)
                        fail_num += 1

                state_action_log[i][-1]["response"] = res
                state_action_log[i][-1]["action"] = eight_phase_list[action_list[i]]

            next_state, _, done, _ = self.env.step(action_list)
            rewards = self.get_norm_reward(next_state)  # my reward

            current_time = self.env.get_current_time()  # in seconds
            state = next_state

            # calculate logger results
            total_reward += sum(rewards)
            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            queue_length_episode.append(sum(queue_length_inter))
            print("Fail Num:", fail_num, "Queuing Vehicles:", sum(queue_length_episode))

            # waiting time
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

        # wandb logger
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else self.dic_traffic_env_conf["RUN_COUNTS"]
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

        results = {
            "test_reward_over": total_reward,
            "test_avg_queue_len_over": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_queuing_vehicle_num_over": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_waiting_time_over": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_travel_time_over": total_travel_time}
        logger.log(results)
        print("Test Round:", test_round, results)
        f_state_action = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        dump_json(state_action_log, f_state_action)
        print("Testing time: ", time.time() - start_time)

        self.env.batch_log_2()

        return results

    def train_test(self):
        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)
        logger = wandb.init(
            project=self.dic_traffic_env_conf['PROJECT_NAME'],
            group=f"{self.dic_traffic_env_conf['MODEL_NAME']}-{self.roadnet}-{self.trafficflow}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases",
            name=f"{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}",
            config=all_config,
        )

        self.test(logger, 0)
        wandb.finish()

    '''
    ======================= Class Utils =======================
    '''
    def get_vehicle_num(self, states):
        veh_nums = []

        for i in range(len(states)):
            vehicle_num = 0

            for lane in states[i]:
                vehicle_num += states[i][lane]['queue_len']
                for cell in range(len(states[i][lane]['cells'])):
                    vehicle_num += states[i][lane]['cells'][cell]

            veh_nums.append(vehicle_num)

        return veh_nums

    def get_norm_reward(self, state):
        rewards = []

        for i in range(len(state)):
            vehicle_num = 0
            queue_length = 0

            intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
            roads = deepcopy(intersection["roads"])
            statistic_state, _, _ = get_state_detail(roads, self.env)
            for lane in statistic_state:
                queue_length += statistic_state[lane]['queue_len']

                vehicle_num += statistic_state[lane]['queue_len']
                for cell in range(len(statistic_state[lane]['cells'])):
                    vehicle_num += statistic_state[lane]['cells'][cell]

            reward = -(queue_length / vehicle_num) if vehicle_num > 0.0 else -0.0
            rewards.append(reward)

        return rewards
