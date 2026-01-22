from minerl.herobraine.hero.handlers.agent.observations.compass import CompassObservation
from minerl.herobraine.hero.handlers.agent.observations.inventory import FlatInventoryObservation
from minerl.herobraine.hero.handlers.agent.observations.equipped_item import _TypeObservation
from minerl.herobraine.hero.handlers.agent.action import ItemListAction
def test_merge_flat_inventory_observation():
    assert FlatInventoryObservation(['stone', 'sandstone', 'lumber', 'wood', 'iron_bar']) | FlatInventoryObservation(['ice', 'ice', 'ice', 'ice', 'ice', 'water']) == FlatInventoryObservation(['stone', 'sandstone', 'lumber', 'wood', 'iron_bar', 'ice', 'water'])