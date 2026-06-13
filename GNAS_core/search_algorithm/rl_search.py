import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from GNAS_core.search_algorithm import search_algorithm_utils as utils
from GNAS_core.search_algorithm.perform_search import estimation
from GNAS_core.model.logger import logger_path


class ArchitectureController(nn.Module):
    """Sequential policy for GNN architecture search."""

    def __init__(self, space_sizes, dataset_feature_dim=4, hidden_dim=64):
        super().__init__()
        self.space_sizes = space_sizes
        self.dataset_feature_dim = dataset_feature_dim

        self.heads = nn.ModuleList()
        input_dim = dataset_feature_dim
        for size in space_sizes:
            self.heads.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, size),
                )
            )
            input_dim += size

    def _build_input(self, dataset_features, one_hots):
        features = dataset_features.unsqueeze(0) if dataset_features.dim() == 1 else dataset_features
        if not one_hots:
            return features
        return torch.cat([features, torch.cat(one_hots, dim=-1)], dim=-1)

    def sample(self, dataset_features):
        one_hots = []
        log_probs = []
        entropies = []
        actions = []

        for step, head in enumerate(self.heads):
            logits = head(self._build_input(dataset_features, one_hots))
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            one_hots.append(F.one_hot(action, self.space_sizes[step]).float())
            actions.append(int(action.item()))

        log_prob = torch.stack(log_probs).sum()
        entropy = torch.stack(entropies).sum()
        return actions, log_prob, entropy

    def evaluate(self, dataset_features, actions):
        one_hots = []
        log_probs = []
        entropies = []

        for step, head in enumerate(self.heads):
            logits = head(self._build_input(dataset_features, one_hots))
            dist = Categorical(logits=logits)
            action = torch.tensor([actions[step]], device=logits.device, dtype=torch.long)
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            one_hots.append(F.one_hot(action, self.space_sizes[step]).float())

        log_prob = torch.stack(log_probs).sum()
        entropy = torch.stack(entropies).sum()
        return log_prob, entropy


class ArchitectureCritic(nn.Module):
    """Value network V(s): expected architecture reward given dataset features."""

    def __init__(self, dataset_feature_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dataset_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, dataset_features):
        features = dataset_features.unsqueeze(0) if dataset_features.dim() == 1 else dataset_features
        return self.net(features).squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.embeddings = []
        self.rewards = []
        self.old_log_probs = []
        self.old_entropies = []

    def add(self, embedding, reward, old_log_prob, old_entropy):
        self.embeddings.append(embedding)
        self.rewards.append(float(reward))
        self.old_log_probs.append(old_log_prob.detach())
        self.old_entropies.append(old_entropy.detach())

    def __len__(self):
        return len(self.rewards)


class RLSearch(object):

    def __init__(self, search_space, args):
        self.search_space = search_space
        self.args = args
        self.space_dict = search_space.space_dict
        self.stack_gnn_architecture = search_space.stack_gnn_architecture
        self.space_sizes = [len(self.space_dict[name]) for name in self.stack_gnn_architecture]

    def _extract_dataset_features(self, graph_data):
        origin_data = graph_data.origin_data
        num_genes = float(origin_data.shape[0])
        num_cells = float(origin_data.shape[1])
        nonzero_ratio = float(np.count_nonzero(origin_data) / max(origin_data.size, 1))
        tf_num = float(self.args.TF_num)

        features = np.array([
            np.log1p(num_genes) / 12.0,
            np.log1p(num_cells) / 12.0,
            nonzero_ratio,
            tf_num / 40.0,
        ], dtype=np.float32)
        return torch.tensor(features, dtype=torch.float32)

    def _decode(self, embedding):
        return utils.gnn_architecture_embedding_decoder(
            embedding,
            self.space_dict,
            self.stack_gnn_architecture,
        )

    def _embedding_key(self, embedding):
        return tuple(embedding)

    def _architecture_key(self, architecture):
        return str(architecture)

    def _evaluate_architecture_with_data(self, architecture, graph_data, reward_cache):
        cache_key = self._architecture_key(architecture)
        if cache_key in reward_cache:
            cached_reward = reward_cache[cache_key]
            print("Cache hit for architecture {}, reward={:.4f}".format(architecture, cached_reward))
            return cached_reward

        reward = estimation([architecture], self.args, graph_data)[0]
        reward_cache[cache_key] = reward
        return reward

    def _sample_unique_embedding(self, controller, dataset_features, evaluated_keys, max_retry=50):
        device = dataset_features.device
        for _ in range(max_retry):
            actions, log_prob, entropy = controller.sample(dataset_features)
            key = self._embedding_key(actions)
            if key not in evaluated_keys:
                return actions, log_prob, entropy

        embedding = utils.random_generate_gnn_architecture_embedding(
            self.space_dict,
            self.stack_gnn_architecture,
        )
        while self._embedding_key(embedding) in evaluated_keys:
            embedding = utils.random_generate_gnn_architecture_embedding(
                self.space_dict,
                self.stack_gnn_architecture,
            )
        log_prob, entropy = controller.evaluate(dataset_features.to(device), embedding)
        return embedding, log_prob, entropy

    def _update_sharing_population(self, sharing_population, sharing_performance,
                                   new_embeddings, new_performances):
        print(35 * "=", "updating sharing population", 35 * "=")
        print("before sharing_performance:\n", sharing_performance)

        _, top_performance = utils.top_population_select(
            sharing_population,
            sharing_performance,
            top_k=self.args.sharing_num,
        )
        avg_performance = np.mean(top_performance) if top_performance else 0.0

        for embedding, performance in zip(new_embeddings, new_performances):
            if performance >= avg_performance:
                sharing_population.append(embedding)
                sharing_performance.append(performance)

        sharing_population, sharing_performance = utils.top_population_select(
            sharing_population,
            sharing_performance,
            top_k=self.args.sharing_num,
        )
        print("after sharing_performance:\n", sharing_performance)
        return sharing_population, sharing_performance

    def _normalize_advantages(self, advantages):
        if len(advantages) <= 1:
            return advantages
        adv_tensor = torch.tensor(advantages, dtype=torch.float32)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        return adv_tensor.tolist()

    def _ppo_update(self, controller, critic, optimizer, dataset_features, buffer):
        rewards = torch.tensor(buffer.rewards, dtype=torch.float32, device=dataset_features.device)
        old_log_probs = torch.stack(buffer.old_log_probs).to(dataset_features.device)

        value = critic(dataset_features).squeeze()
        advantages = rewards - value.detach()
        advantage_list = self._normalize_advantages(advantages.cpu().tolist())
        advantages = torch.tensor(advantage_list, dtype=torch.float32, device=dataset_features.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for ppo_epoch in range(self.args.rl_ppo_epochs):
            value = critic(dataset_features).squeeze()
            policy_losses = []
            entropies = []

            for idx, embedding in enumerate(buffer.embeddings):
                new_log_prob, new_entropy = controller.evaluate(dataset_features, embedding)
                ratio = torch.exp(new_log_prob - old_log_probs[idx])

                advantage = advantages[idx]
                surr1 = ratio * advantage
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.args.rl_ppo_clip,
                    1.0 + self.args.rl_ppo_clip,
                ) * advantage
                policy_losses.append(-torch.min(surr1, surr2))
                entropies.append(new_entropy)

            policy_loss = torch.stack(policy_losses).mean()
            value_loss = torch.mean((value - rewards) ** 2)
            entropy_bonus = torch.stack(entropies).mean()

            loss = (
                policy_loss
                + self.args.rl_value_coef * value_loss
                - self.args.rl_entropy_coef * entropy_bonus
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(controller.parameters()) + list(critic.parameters()),
                self.args.rl_grad_clip,
            )
            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy_bonus.item())

        num_epochs = max(self.args.rl_ppo_epochs, 1)
        return {
            "policy_loss": total_policy_loss / num_epochs,
            "value_loss": total_value_loss / num_epochs,
            "entropy": total_entropy / num_epochs,
        }

    def _reinforce_update(self, controller, optimizer, dataset_features, embedding,
                          log_prob, entropy, reward, baseline):
        advantage = reward - baseline
        policy_loss = -(advantage * log_prob + self.args.rl_entropy_coef * entropy)
        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), self.args.rl_grad_clip)
        optimizer.step()
        return policy_loss.item(), (
            self.args.rl_baseline_decay * baseline
            + (1.0 - self.args.rl_baseline_decay) * reward
        )

    def search_operator(self, graph_data):
        algorithm = self.args.rl_algorithm
        print(35 * "=", "RL architecture search start ({})".format(algorithm.upper()), 35 * "=")
        self._print_search_speed_settings()

        device = self.args.device
        dataset_features = self._extract_dataset_features(graph_data).to(device)
        controller = ArchitectureController(
            self.space_sizes,
            dataset_feature_dim=dataset_features.numel(),
        ).to(device)
        critic = ArchitectureCritic(dataset_feature_dim=dataset_features.numel()).to(device)

        trainable_params = list(controller.parameters())
        if algorithm == "ppo":
            trainable_params += list(critic.parameters())
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.args.rl_lr,
            weight_decay=self.args.rl_weight_decay,
        )

        path = os.path.join(logger_path, self.args.data_save_name, "search_logger")
        if not os.path.exists(path):
            os.makedirs(path)

        evaluated_embeddings = set()
        reward_cache = {}
        sharing_population = []
        sharing_performance = []

        time_initial = time.time()
        warmup_embeddings = []
        while len(warmup_embeddings) < self.args.rl_warmup_num:
            embedding = utils.random_generate_gnn_architecture_embedding(
                self.space_dict,
                self.stack_gnn_architecture,
            )
            key = self._embedding_key(embedding)
            if key in evaluated_embeddings:
                continue
            warmup_embeddings.append(embedding)

        warmup_architectures = [self._decode(emb) for emb in warmup_embeddings]
        warmup_results = estimation(warmup_architectures, self.args, graph_data)
        for embedding, architecture, performance in zip(warmup_embeddings, warmup_architectures, warmup_results):
            evaluated_embeddings.add(self._embedding_key(embedding))
            reward_cache[self._architecture_key(architecture)] = performance

        sharing_population, sharing_performance = utils.top_population_select(
            warmup_embeddings,
            warmup_results,
            top_k=self.args.sharing_num,
        )
        time_initial = time.time() - time_initial
        utils.experiment_time_save_initial(
            path,
            self.args.data_save_name + "_initial_time.txt",
            time_initial,
        )

        baseline = float(np.mean(warmup_results)) if warmup_results else 0.5
        print("RL warmup finished. Baseline reward:", baseline)
        print("Dataset features:", dataset_features.detach().cpu().numpy())

        time_search_list = []
        epoch_list = []
        metrics_history = []

        for epoch in range(self.args.search_epoch):
            print(35 * "=", "RL search epoch", epoch + 1, 35 * "=")
            time_search = time.time()

            epoch_embeddings = []
            epoch_performances = []
            rollout_buffer = RolloutBuffer()

            for sample_idx in range(self.args.sharing_num):
                embedding, log_prob, entropy = self._sample_unique_embedding(
                    controller,
                    dataset_features,
                    evaluated_embeddings,
                )
                architecture = self._decode(embedding)
                reward = self._evaluate_architecture_with_data(
                    architecture,
                    graph_data,
                    reward_cache,
                )
                evaluated_embeddings.add(self._embedding_key(embedding))

                if algorithm == "ppo":
                    rollout_buffer.add(embedding, reward, log_prob, entropy)
                else:
                    loss_value, baseline = self._reinforce_update(
                        controller,
                        optimizer,
                        dataset_features,
                        embedding,
                        log_prob,
                        entropy,
                        reward,
                        baseline,
                    )
                    metrics_history.append({
                        "epoch": epoch + 1,
                        "sample": sample_idx + 1,
                        "algorithm": "reinforce",
                        "reward": reward,
                        "baseline": baseline,
                        "policy_loss": loss_value,
                        "value_loss": 0.0,
                        "entropy": float(entropy.item()),
                    })

                epoch_embeddings.append(embedding)
                epoch_performances.append(reward)

                print(
                    "RL sample {}: architecture={}, reward={:.4f}".format(
                        sample_idx + 1,
                        architecture,
                        reward,
                    )
                )

            epoch_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
            if algorithm == "ppo" and len(rollout_buffer) > 0:
                epoch_metrics = self._ppo_update(
                    controller,
                    critic,
                    optimizer,
                    dataset_features,
                    rollout_buffer,
                )
                values = critic(dataset_features).item()
                metrics_history.append({
                    "epoch": epoch + 1,
                    "sample": "batch",
                    "algorithm": "ppo",
                    "reward": float(np.mean(epoch_performances)),
                    "baseline": values,
                    "policy_loss": epoch_metrics["policy_loss"],
                    "value_loss": epoch_metrics["value_loss"],
                    "entropy": epoch_metrics["entropy"],
                })
                print(
                    "PPO update: policy_loss={:.4f}, value_loss={:.4f}, entropy={:.4f}, "
                    "critic_value={:.4f}".format(
                        epoch_metrics["policy_loss"],
                        epoch_metrics["value_loss"],
                        epoch_metrics["entropy"],
                        values,
                    )
                )

            sharing_population, sharing_performance = self._update_sharing_population(
                sharing_population,
                sharing_performance,
                epoch_embeddings,
                epoch_performances,
            )

            elapsed = time.time() - time_search
            time_search_list.append(elapsed)
            epoch_list.append(epoch + 1)

            utils.experiment_search_data_save(
                path,
                self.args.data_save_name + "_search_epoch_" + str(epoch + 1) + ".txt",
                sharing_population,
                sharing_performance,
                self.space_dict,
                self.stack_gnn_architecture,
            )
            self._save_rl_metrics(
                path,
                epoch + 1,
                baseline if algorithm == "reinforce" else critic(dataset_features).item(),
                epoch_metrics["policy_loss"],
                epoch_metrics["value_loss"],
                epoch_metrics["entropy"],
                elapsed,
                algorithm,
            )

        best_index = int(np.argmax(sharing_performance))
        best_embedding = sharing_population[best_index]
        best_architecture = self._decode(best_embedding)
        best_performance = sharing_performance[best_index]

        controller_path = os.path.join(path, self.args.data_save_name + "_rl_controller.pt")
        torch.save(
            {
                "algorithm": algorithm,
                "actor_state_dict": controller.state_dict(),
                "critic_state_dict": critic.state_dict() if algorithm == "ppo" else None,
                "space_sizes": self.space_sizes,
                "dataset_features": dataset_features.detach().cpu(),
                "best_architecture": best_architecture,
                "best_performance": best_performance,
            },
            controller_path,
        )

        utils.experiment_time_save(
            path,
            self.args.data_save_name + "_search_time.txt",
            epoch_list,
            time_search_list,
        )

        print("Best GNN Architecture:\n", best_architecture)
        print("Best VAL Performance:\n", best_performance)
        print("RL checkpoint saved to:", controller_path)

    def _print_search_speed_settings(self):
        search_epochs = getattr(self.args, "search_train_epoch", None) or self.args.train_epoch
        search_stop = getattr(self.args, "search_stop_num", 10)
        search_folds = getattr(self.args, "search_cv_folds", 3)
        print(
            "Search-speed settings: search_train_epoch={}, search_stop_num={}, "
            "search_cv_folds={} (final test still uses train_epoch={})".format(
                search_epochs,
                search_stop,
                search_folds,
                self.args.train_epoch,
            )
        )

    def _save_rl_metrics(self, path, epoch, baseline, policy_loss, value_loss,
                         entropy, elapsed, algorithm):
        metrics_path = os.path.join(path, self.args.data_save_name + "_rl_metrics.txt")
        write_header = not os.path.exists(metrics_path)
        with open(metrics_path, "a+") as f:
            if write_header:
                f.write("epoch;algorithm;baseline;policy_loss;value_loss;entropy;elapsed\n")
            f.write(
                "{};{};{};{};{};{};{}\n".format(
                    epoch,
                    algorithm,
                    baseline,
                    policy_loss,
                    value_loss,
                    entropy,
                    elapsed,
                )
            )
