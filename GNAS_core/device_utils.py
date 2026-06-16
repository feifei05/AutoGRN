import torch
import dgl


def _dgl_supports_device(device):
    if device.type == 'cpu':
        return True
    try:
        g = dgl.graph((torch.tensor([0]), torch.tensor([1])), num_nodes=2)
        g.to(device)
        return True
    except dgl.DGLError:
        return False


def resolve_runtime_device(device_id):
    """Resolve torch device and verify DGL can use it when CUDA is requested."""
    if device_id < 0:
        device = torch.device('cpu')
        print('Using CPU for PyTorch and DGL.')
        return device

    if not torch.cuda.is_available():
        print('CUDA is not available for PyTorch. Falling back to CPU.')
        return torch.device('cpu')

    device = torch.device(device_id)
    if _dgl_supports_device(device):
        print('Using GPU {} for PyTorch and DGL (cuda:{}).'.format(device_id, device_id))
        return device

    raise RuntimeError(
        'GPU was requested (--device {}), but the installed DGL is CPU-only.\n'
        'Error: Device API cuda is not enabled. Please install the CUDA build of DGL.\n'
        'Example (match your PyTorch CUDA version):\n'
        '  pip uninstall -y dgl\n'
        '  pip install dgl==1.0.1+cu116 -f https://data.dgl.ai/wheels/repo.html\n'
        'Current environment: torch={}, torch.cuda.is_available()={}\n'
        'Alternatively, run on CPU with: --device -1'.format(
            device_id,
            torch.__version__,
            torch.cuda.is_available(),
        )
    )


def move_dgl_graph(graph, device):
    if graph is None:
        return None
    return graph.to(device)
