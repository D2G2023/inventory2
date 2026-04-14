from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
from typing import Any


@dataclass(frozen=True)
class EOQParameters:
    annual_demand: float
    ordering_cost: float
    holding_cost: float
    lead_time_days: float
    reorder_point: float
    horizon_days: float
    initial_inventory: float
    order_quantity: float | None = None

    @property
    def daily_demand(self) -> float:
        return self.annual_demand / 365.0

    @property
    def eoq(self) -> float:
        if self.annual_demand <= 0 or self.ordering_cost <= 0 or self.holding_cost <= 0:
            return 0.0
        return math.sqrt(2.0 * self.annual_demand * self.ordering_cost / self.holding_cost)

    @property
    def order_size(self) -> int:
        quantity = self.order_quantity if self.order_quantity is not None else self.eoq
        return max(1, int(round(quantity)))

    @property
    def demand_interval_days(self) -> float:
        if self.daily_demand <= 0:
            return math.inf
        return 1.0 / self.daily_demand


def run_event_driven_simulation(params: EOQParameters) -> dict[str, Any]:
    if params.horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if params.initial_inventory < 0:
        raise ValueError("initial_inventory cannot be negative")
    if params.reorder_point < 0:
        raise ValueError("reorder_point cannot be negative")

    order_size = params.order_size
    event_queue: list[tuple[float, int, str, dict[str, Any]]] = []
    counter = itertools.count()

    def schedule(time: float, event_type: str, payload: dict[str, Any] | None = None) -> None:
        heapq.heappush(event_queue, (time, next(counter), event_type, payload or {}))

    inventory_on_hand = float(params.initial_inventory)
    inventory_position = float(params.initial_inventory)
    orders_outstanding = 0
    demand_served = 0
    demand_lost = 0
    order_count = 0

    timeline: list[dict[str, Any]] = [
        {
            "time_days": 0.0,
            "event": "start",
            "inventory_on_hand": inventory_on_hand,
            "inventory_position": inventory_position,
            "orders_outstanding": orders_outstanding,
        }
    ]
    orders: list[dict[str, Any]] = []

    schedule(params.demand_interval_days, "demand")

    last_event_time = 0.0
    inventory_area = 0.0

    while event_queue:
        event_time, _, event_type, payload = heapq.heappop(event_queue)
        if event_time > params.horizon_days:
            inventory_area += inventory_on_hand * (params.horizon_days - last_event_time)
            last_event_time = params.horizon_days
            break

        inventory_area += inventory_on_hand * (event_time - last_event_time)
        last_event_time = event_time

        if event_type == "demand":
            if inventory_on_hand >= 1:
                inventory_on_hand -= 1
                inventory_position -= 1
                demand_served += 1
                event_name = "demand_served"
            else:
                demand_lost += 1
                event_name = "demand_lost"

            timeline.append(
                {
                    "time_days": event_time,
                    "event": event_name,
                    "inventory_on_hand": inventory_on_hand,
                    "inventory_position": inventory_position,
                    "orders_outstanding": orders_outstanding,
                }
            )

            if inventory_position <= params.reorder_point:
                arrival_time = event_time + params.lead_time_days
                schedule(arrival_time, "replenishment", {"quantity": order_size})
                inventory_position += order_size
                orders_outstanding += order_size
                order_count += 1
                orders.append(
                    {
                        "order_time_days": event_time,
                        "arrival_time_days": arrival_time,
                        "quantity": order_size,
                    }
                )
                timeline.append(
                    {
                        "time_days": event_time,
                        "event": "reorder_placed",
                        "inventory_on_hand": inventory_on_hand,
                        "inventory_position": inventory_position,
                        "orders_outstanding": orders_outstanding,
                    }
                )

            next_demand = event_time + params.demand_interval_days
            if next_demand <= params.horizon_days:
                schedule(next_demand, "demand")

        elif event_type == "replenishment":
            quantity = int(payload["quantity"])
            inventory_on_hand += quantity
            orders_outstanding -= quantity
            timeline.append(
                {
                    "time_days": event_time,
                    "event": "replenishment_arrived",
                    "inventory_on_hand": inventory_on_hand,
                    "inventory_position": inventory_position,
                    "orders_outstanding": orders_outstanding,
                }
            )

    if last_event_time < params.horizon_days:
        inventory_area += inventory_on_hand * (params.horizon_days - last_event_time)

    average_inventory = inventory_area / params.horizon_days
    fill_rate = demand_served / (demand_served + demand_lost) if (demand_served + demand_lost) else 1.0
    annualized_holding_cost = average_inventory * params.holding_cost
    cycles_per_year = order_count * (365.0 / params.horizon_days)
    annualized_ordering_cost = cycles_per_year * params.ordering_cost
    annual_total_cost = annualized_holding_cost + annualized_ordering_cost

    return {
        "parameters": params,
        "timeline": timeline,
        "orders": orders,
        "summary": {
            "eoq": params.eoq,
            "order_quantity_used": order_size,
            "orders_placed": order_count,
            "demand_served": demand_served,
            "demand_lost": demand_lost,
            "fill_rate": fill_rate,
            "average_inventory": average_inventory,
            "annualized_holding_cost": annualized_holding_cost,
            "annualized_ordering_cost": annualized_ordering_cost,
            "annual_total_cost": annual_total_cost,
        },
    }
