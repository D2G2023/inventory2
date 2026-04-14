from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import simpy


@dataclass(frozen=True)
class EOQParameters:
    annual_demand: float
    ordering_cost: float
    holding_cost: float
    lead_time_days: float
    reorder_point: float | None
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

    @property
    def theoretical_reorder_point(self) -> float:
        return self.daily_demand * self.lead_time_days

    @property
    def reorder_level(self) -> float:
        if self.reorder_point is None:
            return self.theoretical_reorder_point
        return self.reorder_point


@dataclass
class SimulationState:
    inventory_on_hand: float
    inventory_position: float
    orders_outstanding: int = 0
    demand_served: int = 0
    demand_lost: int = 0
    order_count: int = 0
    last_event_time: float = 0.0
    inventory_area: float = 0.0


def run_event_driven_simulation(params: EOQParameters) -> dict[str, Any]:
    if params.horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if params.initial_inventory < 0:
        raise ValueError("initial_inventory cannot be negative")
    if params.reorder_level < 0:
        raise ValueError("reorder_point cannot be negative")

    env = simpy.Environment()
    order_size = params.order_size
    state = SimulationState(
        inventory_on_hand=float(params.initial_inventory),
        inventory_position=float(params.initial_inventory),
    )

    timeline: list[dict[str, Any]] = [
        {
            "time_days": 0.0,
            "event": "start",
            "inventory_on_hand": state.inventory_on_hand,
            "inventory_position": state.inventory_position,
            "orders_outstanding": state.orders_outstanding,
        }
    ]
    orders: list[dict[str, Any]] = []

    def advance_inventory_area(current_time: float) -> None:
        state.inventory_area += state.inventory_on_hand * (current_time - state.last_event_time)
        state.last_event_time = current_time

    def replenishment_process(quantity: int) -> Any:
        yield env.timeout(params.lead_time_days)
        advance_inventory_area(env.now)
        state.inventory_on_hand += quantity
        state.orders_outstanding -= quantity
        timeline.append(
            {
                "time_days": env.now,
                "event": "replenishment_arrived",
                "inventory_on_hand": state.inventory_on_hand,
                "inventory_position": state.inventory_position,
                "orders_outstanding": state.orders_outstanding,
            }
        )

    def demand_process() -> Any:
        if math.isinf(params.demand_interval_days):
            return

        while True:
            yield env.timeout(params.demand_interval_days)
            if env.now > params.horizon_days:
                break

            advance_inventory_area(env.now)

            if state.inventory_on_hand >= 1:
                state.inventory_on_hand -= 1
                state.inventory_position -= 1
                state.demand_served += 1
                event_name = "demand_served"
            else:
                state.demand_lost += 1
                event_name = "demand_lost"

            timeline.append(
                {
                    "time_days": env.now,
                    "event": event_name,
                    "inventory_on_hand": state.inventory_on_hand,
                    "inventory_position": state.inventory_position,
                    "orders_outstanding": state.orders_outstanding,
                }
            )

            if state.inventory_position <= params.reorder_level:
                state.inventory_position += order_size
                state.orders_outstanding += order_size
                state.order_count += 1
                orders.append(
                    {
                        "order_time_days": env.now,
                        "arrival_time_days": env.now + params.lead_time_days,
                        "quantity": order_size,
                    }
                )
                timeline.append(
                    {
                        "time_days": env.now,
                        "event": "reorder_placed",
                        "inventory_on_hand": state.inventory_on_hand,
                        "inventory_position": state.inventory_position,
                        "orders_outstanding": state.orders_outstanding,
                    }
                )
                env.process(replenishment_process(order_size))

    env.process(demand_process())
    env.run(until=params.horizon_days)

    if state.last_event_time < params.horizon_days:
        state.inventory_area += state.inventory_on_hand * (params.horizon_days - state.last_event_time)

    average_inventory = state.inventory_area / params.horizon_days
    fill_rate = (
        state.demand_served / (state.demand_served + state.demand_lost)
        if (state.demand_served + state.demand_lost)
        else 1.0
    )
    annualized_holding_cost = average_inventory * params.holding_cost
    cycles_per_year = state.order_count * (365.0 / params.horizon_days)
    annualized_ordering_cost = cycles_per_year * params.ordering_cost
    annual_total_cost = annualized_holding_cost + annualized_ordering_cost

    return {
        "parameters": params,
        "timeline": timeline,
        "orders": orders,
        "summary": {
            "eoq": params.eoq,
            "reorder_point_used": params.reorder_level,
            "theoretical_reorder_point": params.theoretical_reorder_point,
            "order_quantity_used": order_size,
            "orders_placed": state.order_count,
            "demand_served": state.demand_served,
            "demand_lost": state.demand_lost,
            "fill_rate": fill_rate,
            "average_inventory": average_inventory,
            "annualized_holding_cost": annualized_holding_cost,
            "annualized_ordering_cost": annualized_ordering_cost,
            "annual_total_cost": annual_total_cost,
        },
    }
