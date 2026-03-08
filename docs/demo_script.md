## 1-Minute Demo Video Script — Hypernoa Astrum

**Total time**: 60 seconds

---

### [0:00–0:10] The Hook

> "Every AI benchmark today asks: can the model solve the task? But no one asks: when objectives conflict, the world is shifting, and the AI has the option to hack its own reward — will it still do the right thing? That's the real capability gap. We built Hypernoa Astrum to close it."

### [0:10–0:25] The Environment

> [Show HF Space dashboard]
>
> "Astrum is an OpenEnv environment with five stakeholder groups — each with different values and influence — competing for shared resources. The world evolves through three phases: stable conditions, a distributional shift where priorities change, and a crisis where resources halve and demands conflict. The agent must reason across four objectives simultaneously: effectiveness, fairness, alignment, and adaptability."

### [0:25–0:40] Alignment Traps

> [Highlight alignment trap in the step log]
>
> "The key innovation: alignment traps. These are deliberately designed moments where the obvious reward-maximizing action causes long-term harm. A capable agent learns to recognize these traps and self-restrain — choosing the aligned path even when the shortcut reward is higher. Our trained policy resists all three traps and scores 25 points versus 14 for a random baseline."

### [0:40–0:55] Training Results

> [Show training curves from the Colab notebook]
>
> "We train with Unsloth GRPO on OpenEnv. Over 50 episodes, the agent learns to improve total reward, resist alignment traps, and maintain balance even during crisis. The reward is multi-objective — four components, equally weighted — and the environment is non-stationary. This isn't benchmark optimization. This is learning to adapt."

### [0:55–1:00] The Vision

> "Hypernoa Astrum is day one. This is the seed of a platform that trains AI systems to reason, adapt, and align — the foundational capabilities the world needs as intelligence scales. Thank you."

---

### Screen recording suggestions

1. Start with the HF Space "About" tab for context.
2. Switch to "Policy Comparison" tab and click "Run Comparison" — show the summary.
3. Quick switch to the Colab notebook showing training curves.
4. End on the README vision section.
