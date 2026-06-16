import os

_default_logger_path = os.path.split(os.path.realpath(__file__))[0][:-(7 + len('GNAS_core'))] + "/logger"
logger_path = _default_logger_path


def resolve_logger_path(logger_dir=None):
    if logger_dir is None:
        return logger_path
    return os.path.abspath(logger_dir)


def _gnn_logger_file(data_save_name, logger_dir=None):
    base = resolve_logger_path(logger_dir)
    return os.path.join(base, str(data_save_name), str(data_save_name) + "_gnn_logger.txt")


def gnn_architecture_performance_save(gnn_architecture, performance, data_save_name, logger_dir=None):
    log_file = _gnn_logger_file(data_save_name, logger_dir)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a+", encoding="utf-8") as f:
        f.write(str(gnn_architecture) + ":" + str(performance) + "\n")

    print("gnn architecture and performance save")
    print("save path: ", log_file)
    print(50 * "=")


def gnn_architecture_performance_load(data_save_name, logger_dir=None):
    gnn_architecture_list = []
    performance_list = []

    log_file = _gnn_logger_file(data_save_name, logger_dir)
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() == "":
                continue
            arch_str, perf_str = line.strip().split(":", 1)
            gnn_architecture_list.append(eval(arch_str))
            performance_list.append(eval(perf_str))

    return gnn_architecture_list, performance_list


def load_ranked_architectures(data_save_name, logger_dir=None, top_k=None):
    """Load architectures ranked by search-phase validation AUC (descending)."""
    gnn_architecture_list, performance_list = gnn_architecture_performance_load(
        data_save_name,
        logger_dir=logger_dir,
    )

    arch_perf = {}
    for architecture, performance in zip(gnn_architecture_list, performance_list):
        arch_perf[str(architecture)] = (architecture, float(performance))

    ranked = sorted(arch_perf.values(), key=lambda item: item[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked
