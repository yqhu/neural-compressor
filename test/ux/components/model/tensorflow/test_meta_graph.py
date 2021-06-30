# -*- coding: utf-8 -*-
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test Meta Graph Model."""

import unittest
from unittest.mock import MagicMock, patch

from lpot.ux.components.model.tensorflow.meta_graph import MetaGraphModel


class TestMetaGraphModel(unittest.TestCase):
    """Test MetaGraphModel class."""

    def setUp(self) -> None:
        """Prepare environment."""
        super().setUp()

        get_model_type_patcher = patch(
            "lpot.ux.components.model.model_type_getter.lpot_get_model_type",
        )
        self.addCleanup(get_model_type_patcher.stop)
        get_model_type_mock = get_model_type_patcher.start()
        get_model_type_mock.side_effect = self._get_model_type

        lpot_tensorflow_model_patcher = patch(
            "lpot.ux.components.model.tensorflow.model.LpotTensorflowModel",
        )
        self.addCleanup(lpot_tensorflow_model_patcher.stop)
        lpot_model_instance_mock = lpot_tensorflow_model_patcher.start()
        lpot_model_instance_mock.return_value.input_node_names = [
            "first input node",
            "second input node",
        ]
        lpot_model_instance_mock.return_value.output_node_names = [
            "first output node",
            "second output node",
        ]

    def _get_model_type(self, path: str) -> str:
        """Return model type for well known paths."""
        if "/path/to/meta_graph/" == path:
            return "checkpoint"
        raise ValueError()

    def test_get_framework_name(self) -> None:
        """Test getting correct framework name."""
        self.assertEqual("tensorflow", MetaGraphModel.get_framework_name())

    def test_supports_correct_path(self) -> None:
        """Test getting correct framework name."""
        self.assertTrue(MetaGraphModel.supports_path("/path/to/meta_graph/"))

    def test_supports_incorrect_path(self) -> None:
        """Test getting correct framework name."""
        self.assertFalse(MetaGraphModel.supports_path("/path/to/model.txt"))

    @patch("lpot.ux.components.model.tensorflow.model.check_module")
    def test_guard_requirements_installed(self, mocked_check_module: MagicMock) -> None:
        """Test guard_requirements_installed."""
        model = MetaGraphModel("/path/to/meta_graph/")

        model.guard_requirements_installed()

        mocked_check_module.assert_called_once_with("tensorflow")

    def test_get_input_nodes(self) -> None:
        """Test getting input nodes."""
        model = MetaGraphModel("/path/to/meta_graph/")
        self.assertEqual([], model.get_input_nodes())

    def test_get_output_nodes(self) -> None:
        """Test getting output nodes."""
        model = MetaGraphModel("/path/to/meta_graph/")
        self.assertEqual([], model.get_output_nodes())

    def test_get_input_and_output_nodes(self) -> None:
        """Test getting input nodes."""
        model = MetaGraphModel("/path/to/meta_graph/")
        self.assertEqual([], model.get_input_nodes())
        self.assertEqual([], model.get_output_nodes())


if __name__ == "__main__":
    unittest.main()
