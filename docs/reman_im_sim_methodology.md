# Methodology: EV Motor Remanufacturing Inventory Simulation

## 1. Overview

This document describes the simulation methodology used in `notebooks/borg_reman_sim.ipynb` to study inventory management for a **closed-loop remanufacturing system** for EV motors. The goal is to evaluate replenishment policies under realistic stochastic conditions — variable core returns, uncertain remanufacturing yield, and random supplier lead times — rather than relying on deterministic analytical models such as EOQ.

The simulation is built with **SimPy**, a Python discrete-event simulation (DES) framework. A parametric (s, S) replenishment policy is tested, with results assessed via service level, cost breakdown, and inventory trajectory.

---

## 2. System Description

The system models a **closed-loop supply chain** with two distinct inflow channels:

```
Customer demand
      │
      ▼
  Fulfillment ──────────────────────────────► (stock depleted)
      │
      │  fraction return_rate of fulfilled units
      ▼
  Core returns (Binomial draw)
      │
      │  reman_lead_time weeks later
      ▼
  Remanufacturing yield (Binomial draw)
      │
      └──────────────────────────────────────► (stock replenished)

  Supplier order (when inventory position ≤ ROP)
      │  Normal(lt_mean, lt_std) lead time
      └──────────────────────────────────────► (stock replenished)
```

### Two replenishment channels

| Channel | Trigger | Lead time | Yield uncertainty |
|---|---|---|---|
| Remanufacturing | Automatic after each sale | Fixed `reman_lead_time` weeks | Binomial(n, `reman_success_rate`) |
| External supplier | (s, S) policy review weekly | Normal(`lt_mean`, `lt_std`) | Deterministic (full order arrives) |

---

## 3. Parameters

All parameters are defined in the `Params` dataclass:

### Inventory levels

| Parameter | Default | Description |
|---|---|---|
| `initial_stock` | 60 | Starting stock on hand |
| `max_stock` | 60 | Order-up-to level S |
| `reorder_point` | 35 | Reorder trigger level s |

### Demand

| Parameter | Default | Description |
|---|---|---|
| `demand_per_week` | 25 | Constant weekly customer demand |

### Closed-loop rates

| Parameter | Default | Description |
|---|---|---|
| `return_rate` | 0.87 | Fraction of customers who return old cores |
| `reman_success_rate` | 0.90 | Fraction of returned cores successfully remanufactured |
| `reman_lead_time` | 2.0 weeks | Fixed time to complete a reman batch |

The **net average reman contribution per week** is:

```
net_reman = demand × return_rate × reman_success_rate
          = 25 × 0.87 × 0.90 ≈ 19.6 units/week
```

This leaves an average external supplier need of approximately **5.4 units/week**.

### Supplier replenishment

| Parameter | Default | Description |
|---|---|---|
| `review_period` | 1.0 week | How often inventory position is reviewed |
| `lt_mean` | 2.0 weeks | Mean supplier lead time |
| `lt_std` | 0.5 weeks | Std dev of supplier lead time |

### Cost parameters

| Parameter | Default | Description |
|---|---|---|
| `purchase_cost_per_unit` | 400.00 | Unit cost from supplier |
| `order_cost_per_order` | 250.00 | Fixed cost per order event |
| `holding_cost_per_unit_week` | 4.00 | Cost to hold one unit for one week |

### Simulation horizon

| Parameter | Default | Description |
|---|---|---|
| `sim_weeks` | 104 | Simulation duration (2 years) |

---

## 4. Simulation Engine

### 4.1 Discrete-Event Simulation with SimPy

The simulation uses **SimPy's event-driven paradigm**. Time advances in discrete jumps to the next scheduled event rather than in fixed clock ticks. Four concurrent SimPy processes run inside the `IMSystem` class:

#### Process 1 — `weekly_demand`

Fires every 1 week. Steps:
1. Compute `fulfilled = min(demand, stock)` and `short = demand - fulfilled`.
2. Reduce `stock` by `fulfilled`; log any stockout.
3. Draw core returns: `cores_back ~ Binomial(fulfilled, return_rate)`.
4. For each batch of returned cores, immediately spawn a `reman_process` coroutine.

#### Process 2 — `reman_process(n_cores)`

Launched per return batch. Steps:
1. Wait `reman_lead_time` weeks.
2. Draw good units: `good_units ~ Binomial(n_cores, reman_success_rate)`.
3. Add `good_units` to `stock`; update `reman_units_in_process` counter.

#### Process 3 — `reorder_monitor`

Fires every `review_period` (1 week). Steps:
1. Compute current `inventory_position` (see Section 4.2).
2. If `inventory_position ≤ reorder_point`, place a supplier order for:
   ```
   order_qty = max_stock − inventory_position
   ```
3. Increment `supplier_units_on_order`; spawn a `supplier_order` coroutine.

#### Process 4 — `supplier_order(qty)`

Launched per order event. Steps:
1. Draw lead time: `lead_time ~ max(0.1, Normal(lt_mean, lt_std))`.
2. Wait `lead_time` weeks.
3. Add `qty` to `stock`; decrement `supplier_units_on_order`.

### 4.2 Pipeline-Aware Inventory Position

A key design choice is that the ordering policy uses **pipeline inventory** rather than bare stock-on-hand. This avoids over-ordering during periods when supplier or reman replenishments are already in transit:

```
inventory_position = stock
                   + supplier_units_on_order
                   + expected_reman_pipeline
```

where:

```
expected_reman_pipeline = reman_units_in_process × reman_success_rate
```

Using the expected yield (rather than the raw WIP count) accounts for reman uncertainty without requiring the policy to simulate future outcomes.

### 4.3 Stochastic Draws

All random variables use NumPy's `default_rng` generator, seeded for reproducibility:

| Variable | Distribution | Parameters |
|---|---|---|
| Core returns | Binomial | n = fulfilled units, p = `return_rate` |
| Reman yield | Binomial | n = cores in batch, p = `reman_success_rate` |
| Supplier lead time | Normal (clipped at 0.1) | μ = `lt_mean`, σ = `lt_std` |

---

## 5. Replenishment Policy

The policy is a continuous-review **(s, S)** policy implemented via a periodic monitor:

- **Review period**: every 1 week
- **Reorder point s**: `reorder_point` = 35
- **Order-up-to level S**: `max_stock` = 60
- **Order quantity**: S − inventory\_position (variable, not fixed)

The monitor compares `inventory_position` against `s`. When triggered, it orders enough to bring the *pipeline-adjusted* position up to S. This is sometimes called a **(s, S) order-up-to** or **base-stock** policy variant.

---

## 6. Cost Model

Three cost components are tracked over the simulation horizon:

```
Total Cost = Purchase Cost + Ordering Cost + Holding Cost
```

| Component | Formula |
|---|---|
| Purchase cost | `total_supplier_units_delivered × purchase_cost_per_unit` |
| Ordering cost | `number_of_orders × order_cost_per_order` |
| Holding cost | `avg_stock_on_hand × holding_cost_per_unit_week × sim_weeks` |

Note: remanufacturing costs are not modelled explicitly — they are treated as sunk or handled upstream. Only external supplier procurement drives the purchase and ordering costs here.

---

## 7. Key Performance Indicators (KPIs)

Each simulation run produces the following KPIs:

| KPI | Definition |
|---|---|
| **Service level** | `filled_demand / total_demand` |
| **Total stockouts** | Sum of unmet units across all stockout events |
| **Stockout events** | Count of weeks with any unmet demand |
| **Avg stock on hand** | Time-average of `stock` snapshots |
| **Avg inventory position** | Time-average of pipeline-aware position |
| **Avg expected reman WIP** | Time-average of `reman_units_in_process × reman_success_rate` |
| **Supplier orders placed** | Count of order events |
| **Avg order quantity** | Mean units per supplier order |
| **Total cost** | PC + OC + HC |

---

## 8. Single-Run Results (Baseline)

Baseline parameters, seed = 42, 104 weeks:

| KPI | Value |
|---|---|
| Service level | 86.6% |
| Total stockouts | 345 units (60 events) |
| Reman units restocked | 1,728 |
| Supplier units delivered | 442 |
| Supplier orders placed | 17 |
| Avg stock on hand | 11.2 units |
| Avg inventory position | 41.8 units |
| Avg expected reman WIP | 23.4 units |
| Purchase cost | $176,800 |
| Ordering cost | $4,250 |
| Holding cost | $4,648 |
| **Total cost** | **$185,698** |

The large share of reman inflow (1,728 vs 442 supplier units) confirms that closed-loop returns carry most of the replenishment load. The relatively low service level (86.6%) at the baseline reorder point of 35 suggests the policy needs tuning — likely a higher ROP — to reduce the 60 stockout events.

---

## 9. Visualization

The single-run plot (`inventory_run.png`) has three panels:

1. **Inventory state**: stock on hand and inventory position over time, with reorder point and order-up-to reference lines, supplier delivery markers, and shaded stockout weeks.
2. **Restocking events**: per-week bar chart distinguishing reman inflows (green) from supplier inflows (orange).
3. **Cumulative flows**: cumulative demand vs cumulative reman and supplier inflows, showing the relative contribution of each channel over the horizon.

---

## 10. Monte Carlo Extension

The notebook intro mentions Monte Carlo evaluation of policy parameters. The natural extension is to run `N` replications across different seeds for a given `(reorder_point, max_stock)` pair, then aggregate KPIs to obtain distributions rather than single-run point estimates. This allows:

- Confidence intervals on service level and total cost.
- Policy comparison: sweeping `reorder_point` from, say, 25 to 55 to find the Pareto frontier between cost and service level.
- Sensitivity analysis on `return_rate` and `reman_success_rate` to quantify how much closed-loop variability affects the required safety stock.

The `run(params, seed)` function is already designed for this — each call is independent and returns a complete `IMSystem` log.

---

## 11. Implementation Notes

- **SimPy version**: processes are Python generators using `yield env.timeout(...)`. State is mutated in-place on the shared `IMSystem` object.
- **Snapshot logging (`_snap`)**: called after every state-changing event, producing fine-grained time series for plotting.
- **Lead time clipping**: `max(0.1, Normal(...))` prevents zero or negative lead times from collapsing simultaneous events.
- **Integer arithmetic**: `reman_units_in_process` tracks raw core counts (integers); the pipeline contribution uses the expected float value only for the ordering decision, not for actual stock updates.
