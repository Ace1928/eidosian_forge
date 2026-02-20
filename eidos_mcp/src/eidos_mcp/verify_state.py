from eidosian_core import eidosian

from .state import agent, audit, gis, llm, refactor, type_forge


@eidosian()
def main():
    print(f"GIS: {gis}")
    print(f"Audit: {audit}")
    print(f"LLM: {llm}")
    print(f"Agent: {agent}")
    print(f"Refactor: {refactor}")
    print(f"Type: {type_forge}")


if __name__ == "__main__":
    main()
