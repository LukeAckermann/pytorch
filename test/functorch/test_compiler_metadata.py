# Owner(s): ["module: functorch"]

import torch
from torch._functorch._aot_autograd.schemas import (
    CompilerMetadata,
    InputAliasInfo,
    OutputAliasInfo,
    OutputType,
    ViewAndMutationMeta,
)
from torch._guards import TracingContext
from torch.testing._internal.common_utils import TestCase, run_tests


def _make_input_info(
    *, mutates_data: bool = False, keep_input_mutations: bool = False
) -> InputAliasInfo:
    return InputAliasInfo(
        is_leaf=True,
        mutates_data=mutates_data,
        mutates_metadata=False,
        mutations_hidden_from_autograd=False,
        mutations_under_no_grad_or_inference_mode=False,
        mutation_inductor_storage_resize=False,
        mutates_storage_metadata=False,
        mutation_is_shallow_copy_data=False,
        requires_grad=False,
        keep_input_mutations=keep_input_mutations,
    )


def _make_output_info(
    output_type: OutputType, base_idx: int | None
) -> OutputAliasInfo:
    return OutputAliasInfo(
        output_type=output_type,
        raw_type=torch.Tensor,
        base_idx=base_idx,
        dynamic_dims=None,
        requires_grad=False,
        requires_grad_for_backward=False,
        view_meta_sequence=None,
    )


def _make_metadata() -> ViewAndMutationMeta:
    return ViewAndMutationMeta(
        input_info=[
            _make_input_info(mutates_data=True, keep_input_mutations=True),
            _make_input_info(mutates_data=True),
            _make_input_info(),
        ],
        output_info=[
            _make_output_info(OutputType.alias_of_input, 2),
            _make_output_info(OutputType.is_input, 0),
            _make_output_info(OutputType.alias_of_intermediate, 1),
            _make_output_info(OutputType.non_alias, None),
        ],
        num_intermediate_bases=0,
        keep_input_mutations=False,
        traced_tangents=[],
        traced_tangents_descs=[],
        subclass_inp_meta=[],
        subclass_fw_graph_out_meta=[],
        subclass_tangent_meta=[],
    )


class TestCompilerMetadata(TestCase):
    def test_interface_projects_live_mutation_and_alias_indices(self):
        metadata = _make_metadata()
        backend_metadata = CompilerMetadata(metadata)

        self.assertEqual(backend_metadata.mutated_input_indices, (0, 1))
        self.assertEqual(
            backend_metadata.num_mutated_inp_runtime_indices,
            metadata.num_mutated_inp_runtime_indices,
        )
        self.assertEqual(backend_metadata.aliased_input_indices, (2, 0))
        self.assertFalse(hasattr(backend_metadata, "input_info"))
        self.assertFalse(hasattr(backend_metadata, "output_info"))
        self.assertFalse(hasattr(backend_metadata, "num_intermediate_bases"))

        metadata.input_info = [_make_input_info()]
        metadata.output_info = [_make_output_info(OutputType.alias_of_input, 4)]
        self.assertEqual(backend_metadata.mutated_input_indices, ())
        self.assertEqual(backend_metadata.aliased_input_indices, (4,))

    def test_donation_list_replacement_is_visible(self):
        metadata = _make_metadata()
        metadata.bw_donated_idxs = [1]
        backend_metadata = CompilerMetadata(metadata)

        self.assertIs(backend_metadata.bw_donated_idxs, metadata.bw_donated_idxs)
        replacement = []
        metadata.bw_donated_idxs = replacement
        self.assertIs(backend_metadata.bw_donated_idxs, replacement)

    def test_static_input_indices_write_through(self):
        context = TracingContext(None)
        metadata = _make_metadata()
        context._fw_metadata = metadata
        updated_indices = [3, 5]
        backend_metadata = context.compiler_metadata
        if backend_metadata is None:
            raise AssertionError("Expected compiler metadata")

        backend_metadata.static_input_indices = updated_indices

        self.assertIs(metadata.static_input_indices, updated_indices)

    def test_reacquiring_metadata_after_context_switch(self):
        context = TracingContext(None)
        self.assertFalse(hasattr(context, "fw_metadata"))
        first_metadata = _make_metadata()
        first_metadata.static_input_indices = [1]
        context._fw_metadata = first_metadata
        first_backend_metadata = context.compiler_metadata
        if first_backend_metadata is None:
            raise AssertionError("Expected first compiler metadata")

        second_metadata = _make_metadata()
        second_metadata.static_input_indices = [4]
        context._fw_metadata = second_metadata
        second_backend_metadata = context.compiler_metadata
        if second_backend_metadata is None:
            raise AssertionError("Expected second compiler metadata")

        self.assertEqual(first_backend_metadata.static_input_indices, [1])
        self.assertEqual(second_backend_metadata.static_input_indices, [4])
        second_backend_metadata.static_input_indices = [6]
        self.assertEqual(second_metadata.static_input_indices, [6])
        self.assertEqual(first_metadata.static_input_indices, [1])


if __name__ == "__main__":
    run_tests()
