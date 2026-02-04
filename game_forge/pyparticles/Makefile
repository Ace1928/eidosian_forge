.PHONY: run test benchmark clean

run:
	python3 run.py

test:
	export PYTHONPATH=src && pytest tests/

benchmark:
	export PYTHONPATH=src && python3 benchmarks/benchmark_sim.py

clean:
	rm -rf src/pyparticles/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
