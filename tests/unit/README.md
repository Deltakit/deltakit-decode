# Unit Testing

Each of the following tests should only check functionality from a single class or function.

The organisation of these tests should match the module structure of the source. The only imports into the test file should then be from the corresponding module. The exception is imports of common data-types from `deltakit-core`, such as `OrderedDecodingEdges`, `OrderedSyndrome` and `NXDecodingGraph` or common helper functions such as `dem_to_decoding_graph_and_logicals`. Anything else, such as a code class or noise model, should be mocked.

## Running these tests

Use `pixi` to run these tests.

```sh
pixi run tests deltakit-decode\tests\unit
```
