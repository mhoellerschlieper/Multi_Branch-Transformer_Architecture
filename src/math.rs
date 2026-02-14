// math.rs
// Description: Numerical helpers for softmax, cross entropy, gradient computation,
//              gradient clipping, and additional metrics and diagnostics helpers
//              migrated from layer.rs.
//
// History:
// - 2026-02-01: Consolidate numeric helpers into math.rs.
// - 2026-02-04: Add robust gradient clipping helpers (global norm and element-wise),
//              including sanitization of non-finite gradients.
// - 2026-02-14: Migrate metric and diagnostics computations from layer.rs into math.rs
//              to enforce separation of concerns and reduce duplication.
// Author: Marcus Schlieper (ExpChat.ai)

#![allow(warnings)]

use ndarray::Array2;
use std::cmp::Ordering;

/* ------------------------------ existing functions ------------------------------ */

pub fn softmax_rows(a_logits: &Array2<f32>) -> Array2<f32> {
    let mut a_result = a_logits.clone();

    for mut a_row in a_result.rows_mut() {
        let d_max = a_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let v_exp: Vec<f32> = a_row.iter().map(|&x| (x - d_max).exp()).collect();
        let d_sum: f32 = v_exp.iter().sum();

        if !d_sum.is_finite() || d_sum <= 0.0 {
            let d_uniform = 1.0 / (a_row.len() as f32).max(1.0);
            for j in 0..a_row.len() {
                a_row[j] = d_uniform;
            }
            continue;
        }

        for (j, &d_e) in v_exp.iter().enumerate() {
            a_row[j] = d_e / d_sum;
        }
    }

    a_result
}

pub fn cross_entropy_loss_step(a_probs: &Array2<f32>, v_target: &[usize]) -> f32 {
    if a_probs.nrows() == 0 || a_probs.ncols() == 0 || v_target.is_empty() {
        return 0.0;
    }

    let i_rows = a_probs.nrows().min(v_target.len());
    let i_vocab = a_probs.ncols();

    let mut d_loss: f32 = 0.0;
    for i in 0..i_rows {
        let i_tgt = v_target[i];
        if i_tgt >= i_vocab {
            continue;
        }
        let d_p = a_probs[[i, i_tgt]];
        d_loss -= d_p.max(1e-15).ln();
    }

    d_loss / (i_rows as f32).max(1.0)
}

pub fn compute_gradients_step(a_probs: &Array2<f32>, v_target: &[usize]) -> Array2<f32> {
    let mut a_grads = a_probs.clone();

    let i_rows = a_probs.nrows();
    let i_cols = a_probs.ncols();

    if i_rows == 0 || i_cols == 0 || v_target.is_empty() {
        return a_grads;
    }

    let i_eff = i_rows.min(v_target.len());
    let d_batch = (i_eff as f32).max(1.0);

    for i in 0..i_eff {
        let i_tgt = v_target[i];
        if i_tgt < i_cols {
            a_grads[[i, i_tgt]] -= 1.0;
        }
    }

    a_grads.mapv_inplace(|x| x / d_batch);
    a_grads
}

pub fn sanitize_gradients_inplace(a_grads: &mut Array2<f32>) {
    for d in a_grads.iter_mut() {
        if !d.is_finite() {
            *d = 0.0;
        }
    }
}

pub fn clip_gradients_global_norm(a_grads: &mut Array2<f32>, d_max_norm: f32) {
    if d_max_norm <= 0.0 || !d_max_norm.is_finite() {
        return;
    }

    sanitize_gradients_inplace(a_grads);

    let mut d_norm_sq: f32 = 0.0;
    for &d in a_grads.iter() {
        d_norm_sq += d * d;
    }

    if !d_norm_sq.is_finite() {
        a_grads.fill(0.0);
        return;
    }

    let d_norm = d_norm_sq.sqrt();
    if !d_norm.is_finite() || d_norm <= 0.0 {
        return;
    }

    if d_norm > d_max_norm {
        let d_scale = (d_max_norm / d_norm).max(0.0);
        if d_scale.is_finite() && d_scale > 0.0 {
            a_grads.mapv_inplace(|x| x * d_scale);
        } else {
            a_grads.fill(0.0);
        }
    }
}

pub fn clip_gradients_value(a_grads: &mut Array2<f32>, d_clip_value: f32) {
    if d_clip_value <= 0.0 || !d_clip_value.is_finite() {
        return;
    }

    for d in a_grads.iter_mut() {
        if !d.is_finite() {
            *d = 0.0;
            continue;
        }
        if *d > d_clip_value {
            *d = d_clip_value;
        } else if *d < -d_clip_value {
            *d = -d_clip_value;
        }
    }
}

/* ------------------------------ migrated from layer.rs ------------------------------ */

// Sanitizer for scalar values used across diagnostics and stats.
pub fn sanitize_f32(d_x: f32) -> f32 {
    if d_x.is_finite() { d_x } else { 0.0 }
}

pub fn clamp_prob_f32(d_x: f32) -> f32 {
    if !d_x.is_finite() {
        return 0.0;
    }
    if d_x < 0.0 {
        0.0
    } else if d_x > 1.0 {
        1.0
    } else {
        d_x
    }
}

// Entropy in nats for a probability vector.
pub fn entropy_nat_f32(v_p: &[f32]) -> f32 {
    let mut d_h: f32 = 0.0;
    for &p in v_p.iter() {
        let d_p = clamp_prob_f32(p);
        if d_p > 0.0 {
            d_h -= d_p * d_p.max(1e-12).ln();
        }
    }
    sanitize_f32(d_h)
}

// Normalize a nonnegative distribution; if sum invalid, returns uniform.
pub fn normalize_distribution_f32(v_x: &[f32]) -> Vec<f32> {
    if v_x.is_empty() {
        return Vec::new();
    }

    let mut d_sum: f32 = 0.0;
    for &d in v_x.iter() {
        d_sum += sanitize_f32(d).max(0.0);
    }

    if !d_sum.is_finite() || d_sum <= 0.0 {
        let d_u = 1.0 / (v_x.len() as f32).max(1.0);
        return vec![d_u; v_x.len()];
    }

    v_x.iter()
        .map(|&d| sanitize_f32(d).max(0.0) / d_sum)
        .collect()
}

// Gini coefficient for probabilities in [0, 1].
pub fn gini_coefficient_f32(v_p: &[f32]) -> f32 {
    let i_n = v_p.len();
    if i_n == 0 {
        return 0.0;
    }

    let mut v: Vec<f32> = v_p.iter().map(|&x| clamp_prob_f32(x)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let d_n = i_n as f32;
    let mut d_sum: f32 = 0.0;

    for (i, &p) in v.iter().enumerate() {
        let d_i = i as f32;
        let d_weight = (d_n - d_i - 0.5) / d_n.max(1.0);
        d_sum += p * d_weight;
    }

    let d_g = 1.0 - 2.0 * d_sum;
    if d_g.is_finite() { d_g.clamp(0.0, 1.0) } else { 0.0 }
}

pub fn cosine_similarity_f32(v_a: &[f32], v_b: &[f32]) -> f32 {
    if v_a.is_empty() || v_b.is_empty() || v_a.len() != v_b.len() {
        return 0.0;
    }

    let mut d_dot: f32 = 0.0;
    let mut d_na: f32 = 0.0;
    let mut d_nb: f32 = 0.0;

    for i in 0..v_a.len() {
        let d_x = sanitize_f32(v_a[i]);
        let d_y = sanitize_f32(v_b[i]);
        d_dot += d_x * d_y;
        d_na += d_x * d_x;
        d_nb += d_y * d_y;
    }

    let d_den = (d_na.sqrt() * d_nb.sqrt()).max(1e-12);
    let d_cos = d_dot / d_den;

    if d_cos.is_finite() { d_cos.clamp(-1.0, 1.0) } else { 0.0 }
}

// Flatten Array2 into Vec<f32> with sanitization.
pub fn flatten_array2_f32(a_x: &Array2<f32>) -> Vec<f32> {
    a_x.iter().map(|&d| sanitize_f32(d)).collect()
}

// Mean square energy of an activation tensor.
pub fn mean_square_energy_f32(a_x: &Array2<f32>) -> f32 {
    if a_x.len() == 0 {
        return 0.0;
    }

    let mut d_sum: f32 = 0.0;
    let mut d_cnt: f32 = 0.0;

    for &d in a_x.iter() {
        let d_v = sanitize_f32(d);
        d_sum += d_v * d_v;
        d_cnt += 1.0;
    }

    let d_m = d_sum / d_cnt.max(1.0);
    sanitize_f32(d_m)
}

// Coefficient of variation for a vector.
pub fn coeff_of_variation_f32(v_x: &[f32]) -> f32 {
    if v_x.is_empty() {
        return 0.0;
    }

    let mut d_mean: f32 = 0.0;
    for &d in v_x.iter() {
        d_mean += sanitize_f32(d);
    }
    d_mean /= (v_x.len() as f32).max(1.0);

    if !d_mean.is_finite() || d_mean.abs() < 1e-12 {
        return 0.0;
    }

    let mut d_var: f32 = 0.0;
    for &d in v_x.iter() {
        let d_v = sanitize_f32(d);
        let d_diff = d_v - d_mean;
        d_var += d_diff * d_diff;
    }
    d_var /= (v_x.len() as f32).max(1.0);

    let d_std = d_var.sqrt();
    let d_cv = d_std / d_mean.abs();

    if d_cv.is_finite() { d_cv } else { 0.0 }
}

pub fn mean_vec_f32(v_x: &[f32]) -> f32 {
    if v_x.is_empty() {
        return 0.0;
    }
    let mut d_sum: f32 = 0.0;
    for &d in v_x.iter() {
        d_sum += sanitize_f32(d);
    }
    sanitize_f32(d_sum / (v_x.len() as f32).max(1.0))
}

// Top1-Top2 margin of a probability distribution.
pub fn top1_top2_margin_f32(v_p: &[f32]) -> f32 {
    if v_p.is_empty() {
        return 0.0;
    }

    let mut d_top1: f32 = -1.0;
    let mut d_top2: f32 = -1.0;

    for &p in v_p.iter() {
        let d_p = clamp_prob_f32(p);
        if d_p > d_top1 {
            d_top2 = d_top1;
            d_top1 = d_p;
        } else if d_p > d_top2 {
            d_top2 = d_p;
        }
    }

    let d_margin = (d_top1 - d_top2).max(0.0);
    sanitize_f32(d_margin)
}

// Perplexity proxy computed from selected-token probabilities.
pub fn perplexity_from_selected_probs_f32(v_p_sel: &[f32]) -> f32 {
    if v_p_sel.is_empty() {
        return 0.0;
    }

    let mut d_sum_nll: f32 = 0.0;
    for &p in v_p_sel.iter() {
        let d_p = clamp_prob_f32(p).max(1e-12);
        d_sum_nll += -d_p.ln();
    }

    let d_mean_nll = d_sum_nll / (v_p_sel.len() as f32).max(1.0);
    sanitize_f32(d_mean_nll.exp())
}
