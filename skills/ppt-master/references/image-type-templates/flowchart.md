# Type: flowchart

Sequential blocks connected by arrows, showing process / workflow / pipeline. Used for "how-it-works" visuals, onboarding steps, pipeline diagrams, sequential methodologies.

> **What flowchart means inside a PPT block**: the image internally shows 3-6 stages in a clear directional sequence. Unlike `infographic` (parallel zones), flowchart has **direction**. Unlike `framework` (hub + satellites), flowchart has **sequence**.

## 1. Composition skeleton

Three sub-structures:

### Sub-structure 1 — Horizontal flow (most common)

```
   ┌───┐ → ┌───┐ → ┌───┐ → ┌───┐
   │ 1 │   │ 2 │   │ 3 │   │ 4 │
   └───┘   └───┘   └───┘   └───┘
```

### Sub-structure 2 — Vertical flow

```
   ┌───┐
   │ 1 │
   └─┬─┘
     ↓
   ┌───┐
   │ 2 │
   └─┬─┘
     ↓
   ┌───┐
   │ 3 │
   └───┘
```

### Sub-structure 3 — Looping / cyclical

```
   ┌───┐ → ┌───┐
   │ 1 │   │ 2 │
   └───┘   └─┬─┘
     ↑       ↓
   ┌───┐ ← ┌───┐
   │ 4 │   │ 3 │
   └───┘   └───┘
```

| LAYOUT | 3-6 stage blocks with explicit connectors (arrows or lines with arrowheads) showing direction |
| ELEMENTS | Stages visually consistent (same shape, similar size); connectors uniform stroke weight; arrowheads consistent style |
| NEGATIVE SPACE | Generous around the flow — stages don't touch edges, connectors don't crowd |
| DIRECTIONAL CLARITY | The flow direction is unambiguous |

## 2. Text-policy variants

### `text_policy: none`

Each stage contains an iconic symbol only. Stage labels added in SVG overlay below or above the image.

### `text_policy: embedded`

Each stage may contain a short English keyword (≤2 words) inside or beside it.

---

## 3. Fewshot prompt snippets

**Snippet A — vector-illustration + cool-corporate, horizontal 4-stage flow, text_policy: none, 1200×400**

> Clean flat vector illustration flowchart banner. Four rounded-rectangle stages arranged horizontally across the canvas, separated by uniform gaps. Each stage is filled with primary deep navy `#1E3A5F`, with crisp 2px outlines and 8% soft drop shadow. Between each pair of stages, a thin uniform horizontal arrow with a clean triangular arrowhead in secondary `#F8F9FA` or near-black. The third stage (highlighted as the focal stage) has a thin accent gold `#D4AF37` ring around it — under 5% accent area. Each stage contains one simple iconic symbol in white — an input arrow, a gear, a magnifier, an output arrow respectively. Background field is secondary light gray `#F8F9FA`. Composed for a 1200×400 hero band with 14% inner padding. NO text, letters, numbers, or labels — SVG labels added externally. Color values are rendering guidance only.