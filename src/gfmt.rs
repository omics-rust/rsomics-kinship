use std::fmt::Write;

/// C `printf("%.*g")` of an f64, round-half-to-even on the value's exact
/// decimal expansion. plink2 computes KINSHIP in f64 and formats the f64 (not
/// the underlying rational), so the f64's position relative to a decimal tie
/// decides the direction. Rounding the 17-significant-digit (round-trip-exact)
/// expansion reproduces that without scaling error.
pub fn g(value: f64, precision: usize) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let prec = precision.max(1);
    let neg = value < 0.0;
    let raw = format!("{:.*e}", 16, value.abs());
    let (mantissa, exp) = raw.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: Vec<u8> = mantissa
        .bytes()
        .filter(|b| b.is_ascii_digit())
        .map(|b| b - b'0')
        .collect();
    let (mut digits, exp) = round_sig_even(&digits, exp, prec, false);
    while digits.len() < prec {
        digits.push(0);
    }
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if exp < -4 || exp >= prec as i32 {
        render_e(&mut out, &digits, exp);
    } else {
        render_f(&mut out, &digits, exp);
    }
    out
}

/// `%.*g` of the exact rational `num/den`, half-to-even. plink2 rounds the
/// mathematically exact proportion (count/nsnp), not the f64 nearest to it, so
/// dyadic ties land on the even digit; producing the decimal digits by long
/// division reproduces that exactly.
pub fn g_ratio(num: u64, den: u64, precision: usize) -> String {
    if num == 0 {
        return "0".to_string();
    }
    let prec = precision.max(1);
    // Long-divide num/den into prec+1 significant decimal digits, tracking the
    // leading-digit exponent, plus a sticky bit for the remainder.
    let mut digits: Vec<u8> = Vec::with_capacity(prec + 2);
    let mut rem = num % den;
    let lead = num / den;
    let mut exp;
    if lead > 0 {
        let s = lead.to_string();
        exp = s.len() as i32 - 1;
        digits.extend(s.bytes().map(|b| b - b'0'));
    } else {
        exp = -1;
        while rem != 0 && digits.is_empty() {
            rem *= 10;
            let d = (rem / den) as u8;
            rem %= den;
            if d == 0 {
                exp -= 1;
            } else {
                digits.push(d);
            }
        }
    }
    while digits.len() < prec + 1 {
        rem *= 10;
        digits.push((rem / den) as u8);
        rem %= den;
    }
    let sticky = rem != 0;
    let (mut digits, exp) = round_sig_even(&digits, exp, prec, sticky);
    while digits.len() < prec {
        digits.push(0);
    }
    let mut out = String::new();
    if exp < -4 || exp >= prec as i32 {
        render_e(&mut out, &digits, exp);
    } else {
        render_f(&mut out, &digits, exp);
    }
    out
}

/// Round decimal `digits` to `prec` sig figs, half-to-even, with `sticky`
/// flagging a nonzero tail beyond the supplied digits.
fn round_sig_even(digits: &[u8], exp: i32, prec: usize, sticky: bool) -> (Vec<u8>, i32) {
    if digits.len() <= prec {
        return (digits.to_vec(), exp);
    }
    let mut kept: Vec<u8> = digits[..prec].to_vec();
    let next = digits[prec];
    let rest = sticky || digits[prec + 1..].iter().any(|&d| d != 0);
    let round_up = next > 5 || (next == 5 && (rest || kept[prec - 1] % 2 == 1));
    let mut exp = exp;
    if round_up {
        let mut i = prec;
        loop {
            if i == 0 {
                kept.insert(0, 1);
                kept.pop();
                exp += 1;
                break;
            }
            i -= 1;
            if kept[i] == 9 {
                kept[i] = 0;
            } else {
                kept[i] += 1;
                break;
            }
        }
    }
    (kept, exp)
}

fn render_f(out: &mut String, digits: &[u8], exp: i32) {
    if exp < 0 {
        out.push_str("0.");
        for _ in 0..(-exp - 1) {
            out.push('0');
        }
        for &d in digits {
            out.push((b'0' + d) as char);
        }
        strip_fixed(out);
    } else {
        let int_len = (exp + 1) as usize;
        for (i, &d) in digits.iter().enumerate() {
            if i == int_len {
                out.push('.');
            }
            out.push((b'0' + d) as char);
        }
        for _ in digits.len()..int_len {
            out.push('0');
        }
        strip_fixed(out);
    }
}

fn render_e(out: &mut String, digits: &[u8], exp: i32) {
    out.push((b'0' + digits[0]) as char);
    if digits.len() > 1 {
        out.push('.');
        for &d in &digits[1..] {
            out.push((b'0' + d) as char);
        }
    }
    strip_fixed(out);
    let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
}

fn strip_fixed(s: &mut String) {
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_half_even_ties() {
        assert_eq!(g_ratio(483, 1920, 6), "0.251562");
        assert_eq!(g_ratio(501, 1920, 6), "0.260938");
        assert_eq!(g_ratio(459, 1920, 6), "0.239062");
        assert_eq!(g_ratio(0, 1920, 6), "0");
    }

    #[test]
    fn ratio_plain() {
        assert_eq!(g_ratio(1, 4, 6), "0.25");
        assert_eq!(g_ratio(1, 2, 6), "0.5");
        assert_eq!(g_ratio(1929, 1929, 6), "1");
    }

    #[test]
    fn f64_g_matches_c() {
        assert_eq!(g(0.0, 6), "0");
        assert_eq!(g(-0.0214286, 6), "-0.0214286");
        assert_eq!(g(0.5, 6), "0.5");
        assert_eq!(g(f64::NAN, 6), "nan");
        assert_eq!(g(0.00012345678, 6), "0.000123457");
    }
}
