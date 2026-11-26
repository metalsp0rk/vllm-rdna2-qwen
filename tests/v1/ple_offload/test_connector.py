# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the PLE CPU-offload connector (MRV1/MRV2 data paths).

Originally carried inside tests/v1/worker/test_gpu_model_runner.py; moved to
their own module with the v0.34.0 rebase because the connector is
runner-agnostic.
"""

import queue
import threading
from multiprocessing.reduction import ForkingPickler
from unittest.mock import Mock

import pytest
import torch
import torch.multiprocessing as torch_mp
import zmq

import vllm.v1.ple_offload.connector as ple_offload_connector_module
from vllm.v1.ple_offload.connector import PleOffloadConnector

def test_ple_offload_h2h_uses_bound_input_buffers() -> None:
    """Stage MRV1 inputs from allocations bound during connector setup."""
    source_input_ids = torch.tensor([7, -1, 11, -1], dtype=torch.int32)
    source_query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    socket = Mock()
    connector = PleOffloadConnector.__new__(PleOffloadConnector)
    connector.tp_rank = 0
    connector.dp_rank = 0
    connector._input_ids_buf = torch.full((4,), -99, dtype=torch.int32)
    connector._query_start_loc_buf = torch.full((3,), -99, dtype=torch.int32)
    connector._ngram_context_buf = None
    connector._input_ids_source = source_input_ids
    connector._query_start_loc_source = source_query_start_loc
    connector._ngram_context_source = None
    connector._uses_cuda_inputs = False
    connector._validate_input_sources()
    connector._request_queue = queue.Queue(maxsize=1)
    connector._launch_seq = 0
    connector._t_launch = connector._t_wait = connector._t_stage = 0.0

    connector._launch(num_reqs=2, num_tokens=4)

    # launch only queues MRV1 work; the notifier thread performs the H2H copy.
    assert connector._input_ids_buf.tolist() == [-99, -99, -99, -99]
    queued = connector._request_queue.get_nowait()
    assert queued is not None
    request, input_ready, _seq = queued
    connector._process_request(request, input_ready, socket)

    assert connector._input_ids_buf.tolist() == [7, -1, 11, -1]
    assert connector._query_start_loc_buf.tolist() == [0, 2, 4]
    assert source_input_ids.tolist() == [7, -1, 11, -1]
    socket.send.assert_called_once()


def test_ple_offload_request_thread_copies_mrv1_and_stops() -> None:
    """Keep MRV1 H2H and request publication off the model thread."""
    connector = PleOffloadConnector.__new__(PleOffloadConnector)
    connector.tp_rank = 0
    connector.dp_rank = 0
    connector._input_ids_buf = torch.empty(4, dtype=torch.int32)
    connector._query_start_loc_buf = torch.empty(3, dtype=torch.int32)
    connector._ngram_context_buf = None
    connector._input_ids_source = torch.tensor([7, 8, 9, 10], dtype=torch.int32)
    connector._query_start_loc_source = torch.tensor([0, 2, 4], dtype=torch.int32)
    connector._ngram_context_source = None
    connector._uses_cuda_inputs = False
    connector._pinned_input_buffers = []
    connector._d2h_stream = None
    connector._input_ready_event = None
    connector._d2h_done_event = None
    connector._request_queue = queue.Queue(maxsize=1)
    connector._launch_seq = 0
    connector._t_launch = connector._t_wait = connector._t_stage = 0.0
    connector._request_thread = None
    connector._request_thread_ready = threading.Event()
    connector._zmq_ctx = zmq.Context()
    connector._registration_socket = None

    ipc_addr = f"inproc://ple-offload-{id(connector)}"
    pull_socket = connector._zmq_ctx.socket(zmq.PULL)
    pull_socket.setsockopt(zmq.RCVTIMEO, 3000)
    pull_socket.bind(ipc_addr)
    try:
        connector._start_request_thread(ipc_addr)
        connector._launch(num_reqs=2, num_tokens=4)

        assert pull_socket.recv()
        assert connector._input_ids_buf.tolist() == [7, 8, 9, 10]
        assert connector._query_start_loc_buf.tolist() == [0, 2, 4]
    finally:
        pull_socket.close(linger=0)
        connector.close()
        connector.close()

    assert connector._request_thread is None
    assert connector._zmq_ctx is None


def test_ple_offload_request_thread_failure_exits_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat a request-thread failure as a fatal worker error."""
    connector = PleOffloadConnector.__new__(PleOffloadConnector)
    connector._zmq_ctx = None
    connector._request_thread_ready = threading.Event()
    exit_mock = Mock(side_effect=SystemExit(1))
    monkeypatch.setattr(ple_offload_connector_module.os, "_exit", exit_mock)

    with pytest.raises(SystemExit):
        connector._request_loop("inproc://unused")

    exit_mock.assert_called_once_with(1)


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="GPU is required")
def test_ple_offload_mrv2_copies_into_pinned_shared_buffers() -> None:
    """Keep MRV2 D2H asynchronous without a second CPU staging buffer."""
    socket = Mock()
    connector = PleOffloadConnector.__new__(PleOffloadConnector)
    connector.device = torch.device("cuda:0")
    connector.tp_rank = 0
    connector.dp_rank = 0
    original_strategy = torch_mp.get_sharing_strategy()
    try:
        torch_mp.set_sharing_strategy("file_descriptor")
        connector._input_ids_buf = torch.full(
            (4,), -99, dtype=torch.int32
        ).share_memory_()
        connector._query_start_loc_buf = torch.full(
            (3,), -99, dtype=torch.int32
        ).share_memory_()
        connector._ngram_context_buf = torch.full(
            (2, 3), -99, dtype=torch.int32
        ).share_memory_()
        input_buffers = (
            connector._input_ids_buf,
            connector._query_start_loc_buf,
            connector._ngram_context_buf,
        )
        initial_ptrs = tuple(buffer.data_ptr() for buffer in input_buffers)

        torch_mp.set_sharing_strategy("file_system")
        ForkingPickler.dumps(input_buffers)
    finally:
        torch_mp.set_sharing_strategy(original_strategy)
    final_ptrs = tuple(buffer.data_ptr() for buffer in input_buffers)
    assert final_ptrs != initial_ptrs

    connector._pinned_input_buffers = []
    connector._out_bufs = {}  # set by _attach_output_buffers in real init
    connector._request_queue = queue.Queue(maxsize=1)
    connector._launch_seq = 0
    connector._t_launch = connector._t_wait = connector._t_stage = 0.0

    with torch.accelerator.device_index(connector.device.index):
        input_ids = torch.tensor(
            [7, -1, 11, -1], dtype=torch.int32, device=connector.device
        )
        query_start_loc = torch.tensor(
            [0, 2, 4], dtype=torch.int32, device=connector.device
        )
        ngram_context = torch.tensor(
            [[1, 2, 3], [4, 5, 6]],
            dtype=torch.int32,
            device=connector.device,
        )
        connector._input_ids_source = input_ids
        connector._query_start_loc_source = query_start_loc
        connector._ngram_context_source = ngram_context
        connector._uses_cuda_inputs = True
        connector._validate_input_sources()
        connector._pin_input_buffers()
        assert (
            tuple(buffer.data_ptr() for buffer in connector._pinned_input_buffers)
            == final_ptrs
        )
        connector._d2h_stream = torch.cuda.Stream(device=connector.device)
        connector._input_ready_event = torch.cuda.Event()
        connector._d2h_done_event = torch.cuda.Event()
        try:
            connector._launch(num_reqs=2, num_tokens=4)

            # The model thread only records input readiness and queues metadata.
            assert connector._input_ids_buf.tolist() == [-99, -99, -99, -99]
            queued = connector._request_queue.get_nowait()
            assert queued is not None
            request, input_ready, _seq = queued
            connector._process_request(request, input_ready, socket)

            assert connector._input_ids_buf.tolist() == [7, -1, 11, -1]
            assert connector._query_start_loc_buf.tolist() == [0, 2, 4]
            assert connector._ngram_context_buf.tolist() == [
                [1, 2, 3],
                [4, 5, 6],
            ]
            socket.send.assert_called_once()
        finally:
            torch.accelerator.synchronize(connector.device)
            connector._unpin_input_buffers()

