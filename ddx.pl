
main(Formula, Tex) :-
	f_ddx(Formula, Dx),
	f_maxsimple(Dx, Simple),
	f_postsimple(Simple, PSimple),
	phrase(term_latex_root(PSimple), Tex).

main_print(Formula) :-
	main(Formula, Tex),
	format('~s~n', [Tex]).

% Helper libraries & imports
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
:- use_module(library(clpfd)).
:- use_module(library(lambda)).

lastelement_of(S, [S]).
lastelement_of(S, [_|Ss]) :-
	lastelement_of(S, Ss).

filter_list_list(_, [], []).
filter_list_list(S, [S|Xs], Ys) :-
	filter_list_list(S, Xs, Ys).
filter_list_list(S, [X|Xs], [X|Ys]) :-
	dif(S, X),
	filter_list_list(S, Xs, Ys).

before_after_inlist(B, A, [B|Xs]) :-
	member(A, Xs).
before_after_inlist(B, A, [X|Xs]) :-
	dif(B, X),
	before_after_inlist(B, A, Xs).

% Datatypes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mynumber(N) :- rational(N).

% Constants are e, pi and numbers (n).
constant(e).
constant(pi).
constant(n(_)).

constant_value(n(N), N).

variable(x).

binop_type_arg_arg(A+B, '+', A, B).
binop_type_arg_arg(A-B, '-', A, B).
binop_type_arg_arg(A*B, '*', A, B).
binop_type_arg_arg(A/B, '/', A, B).
binop_type_arg_arg(A^B, '^', A, B).

fn_type_arg(ln(A), ln, A).
fn_type_arg(sin(A), sin, A).
fn_type_arg(cos(A), cos, A).
fn_type_arg(tan(A), tan, A).
fn_type_arg(cot(A), cot, A).

term(V) :- variable(V).
term(C) :- constant(C).
term(BO) :- binop_type_arg_arg(BO, _, A, B), term(A), term(B).
term(FN) :- fn_type_arg(FN, _, A), term(A).

or(true, true, true).
or(true, false, true).
or(false, true, true).
or(false, false, false).

term_latex_root(T) -->
	"\\int", term_latex(T, false, true, false), "\\operatorname{d}\\!x".

% arguments: (term, multiplication brace, addition brace, after brace)
term_latex(V, _, _, _) -->
	{ variable(V) },
	variable_latex(V).
term_latex(C, _, _, AfterBr) -->
	{ constant(C) },
	constant_latex(C, AfterBr).
term_latex(A+B, _, AddBr, AfterBr) -->
	obrace(AddBr),
	{ or(AddBr, AfterBr, NeedBr) },
	term_latex(A, false, false, NeedBr),
	" + ",
	term_latex(B, false, false, false),
	cbrace(AddBr).
term_latex(A-B, _, AddBr, AfterBr) -->
	obrace(AddBr),
	{ or(AddBr, AfterBr, NeedBr) },
	term_latex(A, false, false, NeedBr),
	" - ",
	term_latex(B, false, true, false),
	cbrace(AddBr).
term_latex(A*B, MulBr, _, AfterBr) -->
	obrace(MulBr),
	{ or(MulBr, AfterBr, NeedBr) },
	term_latex(A, false, true, NeedBr),
	" ",
	term_latex(B, false, true, false),
	cbrace(MulBr).
term_latex(A/B, _, _, _) -->
	"\\frac{",
	term_latex(A, false, false, true),
	"}{",
	term_latex(B, false, false, true),
	"}".
term_latex(A^B, MulBr, _, _) -->
	{ dif(B, 1/2) },
	obrace(MulBr),
	term_latex(A, true, true, false),
	"^{",
	term_latex(B, false, false, true),
	"}",
	cbrace(MulBr).
term_latex(A^(1/2), _, _, _) -->
	"\\sqrt{",
	term_latex(A, false, false, true),
	"}".
term_latex(sin(T), _, _, _) -->
	"\\sin(", term_latex(T, false, false, true), ")".
term_latex(cos(T), _, _, _) -->
	"\\cos(", term_latex(T, false, false, true), ")".
term_latex(tan(T), _, _, _) -->
	"\\tan(", term_latex(T, false, false, true), ")".
term_latex(cot(T), _, _, _) -->
	"\\cot(", term_latex(T, false, false, true), ")".
term_latex(ln(T), _, _, _) -->
	"\\ln(", term_latex(T, false, false, true), ")".

obrace(false) -->
	"".
obrace(true) -->
	"\\left[".

cbrace(false) -->
	"".
cbrace(true) -->
	"\\right]".

constant_latex(e, _) -->
	"e".
constant_latex(pi, _) -->
	"\\pi".
constant_latex(n(N), _) -->
	{ integer(N) },
	{ N >= 0 },
	{ format(string(Tex), '~d', [N]) },
	Tex.
constant_latex(n(N), false) -->
	{ integer(N) },
	{ N < 0 },
	{ format(string(Tex), '(~d)', [N]) },
	Tex.
constant_latex(n(N), true) -->
	{ integer(N) },
	{ N < 0 },
	{ format(string(Tex), '~d', [N]) },
	Tex.

variable_latex(x) -->
	"x".

% Differentiation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basic rules
f_ddx(V, n(1)) :-
	variable(V).
f_ddx(C, n(0)) :-
    constant(C).
% Core rules: addition, multiplication and power
f_ddx(A+B, AA+BB) :-
    f_ddx(A, AA),
    f_ddx(B, BB).
f_ddx(A*B, (AA*B + A*BB)) :-
    f_ddx(A, AA),
    f_ddx(B, BB).
f_ddx(A^B, (A^(B-n(1))) * (B*AA + A*ln(A)*BB)) :-
    f_ddx(A, AA),
    f_ddx(B, BB).
% Elementary functions
f_ddx(ln(A), A^n(-1) * AA) :-
    f_ddx(A, AA).
f_ddx(sin(A), cos(A) * AA) :-
	f_ddx(A, AA).
f_ddx(cos(A), n(-1) * sin(A) * AA) :-
	f_ddx(A, AA).
% Extra rules: These terms will be differentiated by changing them to ones that
% can be differentiated.
f_ddx(A-B, D) :-
	f_ddx(A + n(-1)*B, D).
f_ddx(A/B, D) :-
	f_ddx(A * B^n(-1), D).
f_ddx(tan(A), D) :-
	f_ddx(sin(A) / cos(A), D).
f_ddx(cot(A), D) :-
	f_ddx(cos(A) / sin(A), D).


:- f_ddx(x, n(1)).
:- f_ddx(n(123), n(0)).
:- f_ddx(n(0), n(0)).
:- f_ddx(+(n(10), x), +(n(0), n(1))).
:- f_ddx(+(x, x), +(n(1), n(1))).
:- f_ddx(*(x, x), +(*(n(1), x), *(x, n(1)))).
:- f_ddx(ln(x), *(^(x, n(-1)), n(1))).
:- f_ddx(ln(*(n(2), x)), *(^(*(n(2), x), n(-1)), +(*(n(0), x), *(n(2), n(1))))). 

% Simplification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Helpers
muldivpow(_*_).
muldivpow(_/_).
muldivpow(_^_).

plusminus(_+_).
plusminus(_-_).

muldivpow_relevancy(A*B, Rs) :-
	!,
	muldivpow_relevancy(A, RA),
	muldivpow_relevancy(B, RB),
	append(RB, RA, Rs).
muldivpow_relevancy(A/B, Rs) :-
	!,
	muldivpow_relevancy(A, RA),
	muldivpow_relevancy(B, RB),
	append(RA, RB, Rs).
muldivpow_relevancy(A^B, Rs) :-
	!,
	muldivpow_relevancy(A, RA),
	muldivpow_relevancy(B, RB),
	append(RA, RB, Rs).
muldivpow_relevancy(A, [A]).

match_replace_list_list_(M, R, [M|Xs], [R|Xs], [f|XsUsed], [t|XsUsed]).
match_replace_list_list_(M, R, [M|Xs], [M|Ys], [t|XsUsed], [t|YsUsed]) :-
	match_replace_list_list_(M, R, Xs, Ys, XsUsed, YsUsed).
match_replace_list_list_(M, R, [X|Xs], [X|Ys], [U|XsUsed], [U|YsUsed]) :-
	dif(M, X),
	match_replace_list_list_(M, R, Xs, Ys, XsUsed, YsUsed).

matches_replaces_list_list_([], [], Xs, Xs, XsUsed, XsUsed).
matches_replaces_list_list_([M|Ms], [R|Rs], Xs, Zs, XsUsed, ZsUsed) :-
	matches_replaces_list_list_(Ms, Rs, Xs, Ys, XsUsed, YsUsed),
	match_replace_list_list_(M, R, Ys, Zs, YsUsed, ZsUsed).

matches_replaces_list_list(Ms, Rs, Xs, Ys) :-
	length(Xs, XsLen),
	length(XsUsed, XsLen),
	maplist(\X^( X=f ), XsUsed),
	matches_replaces_list_list_(Ms, Rs, Xs, Ys, XsUsed, _).

:- matches_replaces_list_list([1,2], [2,1], [1,3,2], [2,3,1]).

% Addidive order
additionorder([PM, ln(_), sin(_), cos(_), tan(_), cot(_), MDP, V, C]) :-
	plusminus(PM),
	muldivpow(MDP),
	variable(V),
	constant(C).

plus_listtermorder([A|_], [B|_]) :-
	plus_termorder(A, B).
plus_listtermorder([A|As], [A|Bs]) :-
	plus_listtermorder(As, Bs).

plus_termorder(A, B) :-
	additionorder(Order),
	before_after_inlist(A, B, Order).
plus_termorder(FN1, FN2) :-
	fn_type_arg(FN1, T, A1),
	fn_type_arg(FN2, T, A2),
	plus_termorder(A1, A2).
plus_termorder(MDP1, MDP2) :-
	muldivpow(MDP1),
	muldivpow(MDP2),
	muldivpow_relevancy(MDP1, Rs1),
	muldivpow_relevancy(MDP2, Rs2),
	plus_listtermorder(Rs1, Rs2).
plus_termorder(e, pi).
plus_termorder(e, n(_)).
plus_termorder(pi, n(_)).
plus_termorder(n(N), n(M)) :-
	N > M.

rel_plus_termorder(>, t(_, A), t(_, B)) :- plus_termorder(A, B), !.
rel_plus_termorder(<, _, _) :- !.

% Multiplicative order
mulorder([C, V, MDP, ln(_), sin(_), cos(_), tan(_), cot(_), PM]) :-
	constant(C),
	variable(V),
	muldivpow(MDP),
	plusminus(PM).

mul_listtermorder([A|_], [B|_]) :-
	mul_termorder(A, B).
mul_listtermorder([A|As], [A|Bs]) :-
	mul_listtermorder(As, Bs).

mul_termorder(A, B) :-
	mulorder(Order),
	before_after_inlist(A, B, Order).
mul_termorder(FN1, FN2) :-
	fn_type_arg(FN1, T, A1),
	fn_type_arg(FN2, T, A2),
	mul_termorder(A1, A2).
mul_termorder(MDP1, MDP2) :-
	muldivpow(MDP1),
	muldivpow(MDP2),
	muldivpow_relevancy(MDP1, Rs1),
	muldivpow_relevancy(MDP2, Rs2),
	mul_listtermorder(Rs1, Rs2).
mul_termorder(n(_), e).
mul_termorder(n(_), pi).
mul_termorder(n(N), n(M)) :-
	N < M.
mul_termorder(e, pi).

rel_mul_termorder(>, t(*, _), t(/, _)) :- !.
rel_mul_termorder(>, t(Op, A), t(Op, B)) :- mul_termorder(A, B), !.
rel_mul_termorder(<, _, _) :- !.

% Collect all terms in a (left assoc.) multiplication chain
% 1*2/3 results in [t(*,1), t(*,2), t(/,3)] - t "means" term
mul_terms(A, [t(*, A)]) :-
	not(A = _*_),
	not(A = _/_).
mul_terms(A/B, [t(/, B)|Us]) :-
	mul_terms(A, Us).
mul_terms(A*B, [t(*, B)|Us]) :-
	mul_terms(A, Us).

terms_mul_([t(/, T)], n(1)/T).
terms_mul_([t(*, T)], T).
terms_mul_([t(/, T)|Ts], R/T) :-
	terms_mul_(Ts, R).
terms_mul_([t(*, T)|Ts], R*T) :-
	terms_mul_(Ts, R).
terms_mul([], n(1)).
terms_mul(Xs, Mul) :-
	terms_mul_(Xs, Mul).

matches_replaces_mul_result(Ms, Rs, T, U) :-
	mul_terms(T, Ts),
	matches_replaces_list_list(Ms, Rs, Ts, Us),
	terms_mul(Us, U).

:- matches_replaces_mul_result([t(*,A), t(*,A)], [t(*,A^2), t(*,1)], 1*2*1, 1^2*2*1).

% Collect all terms in a (left associative) addition chain
% 1+2-3 results in [t(+,1), t(+,2), t(-,3)] - t "means" term
addition_terms(n(-1)*A, [t(-, A)]).
addition_terms(A, [t(+, A)]) :-
	not(A = _+_),
	not(A = _-_),
	not(A = n(-1)*_).
addition_terms(A-B, [t(-, B)|Us]) :-
	addition_terms(A, Us).
addition_terms(A+B, [t(+, B)|Us]) :-
	addition_terms(A, Us).

terms_addition_([t(+, T)], T).
terms_addition_([t(-, n(-1)*T)], n(-1)*T).
terms_addition_([t(-, T)], n(-1)*T) :-
	not(T = n(-1)*_).
terms_addition_([t(-, T)|Ts], R-T) :-
	terms_addition_(Ts, R).
terms_addition_([t(+, T)|Ts], R+T) :-
	terms_addition_(Ts, R).
terms_addition([], n(0)).
terms_addition(Xs, Add) :-
	terms_addition_(Xs, Add).

matches_replaces_addition_result(Ms, Rs, T, U) :-
	addition_terms(T, Ts),
	matches_replaces_list_list(Ms, Rs, Ts, Us),
	terms_addition(Us, U).

:- matches_replaces_addition_result([t(+,A), t(+,A)], [t(+,2*A),t(+,0)], 1+2+1, 2*1+2+0).

% Simplification rules, unfortunately the order is relevant, because later rules
% might invoke a endless loop especially if the + or * associativity is not fixed
% yet. :(

t_optpow(A, A, n(1)).
t_optpow(A^B, A, B).

% Fix associativity
f_simple(A+(B+C), A+B+C).
f_simple(A+(B-C), A+B-C).
f_simple(A-(B+C), A-B-C).
f_simple(A-(B-C), A-B+C).
f_simple(A*(B*C), A*B*C).
f_simple(A*(B/C), A*B/C).
f_simple(A/(B*C), A/B/C).
f_simple(A/(B/C), A/B*C).
% Addition & Subdraction
f_simple(T, U) :-  % remove zeros
	addition_terms(T, Ts),
	member(t(_,n(0)), Ts),
	filter_list_list(t(_,n(0)), Ts, Us),
	terms_addition(Us, U).
f_simple(T, U) :-  % merge constants
	matches_replaces_addition_result([t(+,n(A)), t(+,n(B))], [t(+,n(N)), t(+,n(0))], T, U),
	N is A + B.
f_simple(T, U) :-  % merge constants
	matches_replaces_addition_result([t(+,n(A)), t(-,n(B))], [t(+,n(N)), t(-,n(0))], T, U),
	N is A - B.
f_simple(T, U) :-  % A+A = 2*A
	dif(A, n(0)),
	matches_replaces_addition_result([t(Op,A), t(Op,A)], [t(Op,n(2)*A), t(Op,n(0))], T, U).
f_simple(T, U) :-  % N*A+A = (N+1)*A
	dif(A, n(0)),
	matches_replaces_addition_result([t(Op,n(N)*A), t(Op,A)], [t(Op,n(M)*A), t(Op,n(0))], T, U),
	M is N + 1.
f_simple(T, U) :-  % TODO merge with ^
	dif(A, n(0)),
	matches_replaces_addition_result([t(Op,A*n(N)), t(Op,A)], [t(Op,n(M)*A), t(Op,n(0))], T, U),
	M is N + 1.
% Multiplication & Division
f_simple(T, n(0)) :-  % remove zeros
	mul_terms(T, Ts),
	member(t(*, n(0)), Ts).
f_simple(T, U) :-  % remove ones
	mul_terms(T, Ts),
	member(t(_,n(1)), Ts),
	filter_list_list(t(_,n(1)), Ts, Us),
	terms_mul(Us, U).
f_simple(T, U) :-  % merge constants
	matches_replaces_mul_result([t(Op,n(A)), t(Op,n(B))], [t(Op,n(N)), t(Op,n(1))], T, U),
	N is A * B.
f_simple(T, U) :-  % merge constants
	matches_replaces_mul_result([t(*,n(A)), t(/,n(B))], [t(*,n(N)), t(*,n(1))], T, U),
	N is A / B,
	integer(N).
f_simple(T, U) :-  % merge constants (gcd)
	matches_replaces_mul_result([t(*,n(A)), t(/,n(B))], [t(*,n(C)), t(/,n(D))], T, U),
	N is A rdiv B,
	C rdiv D = N.
f_simple(T, U) :-  % A*A = A^2
	dif(A, n(1)),
	dif(A, n(0)),
	matches_replaces_mul_result([t(*,A), t(*,A)], [t(*,A^n(2)), t(*,n(1))], T, U).
f_simple(T, U) :-  % A/A = 1
	dif(A, n(1)),
	dif(A, n(0)),
	matches_replaces_mul_result([t(*,A), t(/,A)], [t(*,n(1)), t(/,n(1))], T, U).
% Power
f_simple(_^n(0), n(1)).
f_simple(A^n(1), A).
f_simple(n(1)^_, n(1)).
f_simple(n(0)^_, n(0)).
f_simple((A^B)^C, A^(B*C)).
f_simple(n(A)^n(B), N) :-  % merge constants
	N is A ^ B.
f_simple(A^n(N), n(1)/(A^n(M))) :-  % A^-1 = 1/A
	N < 0,
	M is (-1) * N.
f_simple(T, U) :-  % A^N * A = A^(N+1)
	matches_replaces_mul_result([t(*,A^N), t(*,A)], [t(*,A^(N+n(1))), t(*,n(1))], T, U).
f_simple(T, U) :-  % A^B * A^C = A^(B+C)
	matches_replaces_mul_result([t(*,A^B), t(*,A^C)], [t(*,A^(B+C)), t(*,n(1))], T, U).
f_simple(T, U) :-  % A^N / A^M = A^(N-M)
	dif(A, n(0)),
	dif(A, n(1)),
	matches_replaces_mul_result([t(*,A^N), t(/,A^M)], [t(*,A^(N-M)), t(/,n(1))], T, U).
f_simple(T, U) :-  % A / A^N = 1 / A^(N-1)
	dif(A, n(0)),
	dif(A, n(1)),
	matches_replaces_mul_result([t(*,A), t(/,A^N)], [t(*,n(1)), t(/, A^(N-n(1)))], T, U).
f_simple(T, U) :-  % A^N / A = A^(N-1)
	dif(A, n(0)),
	dif(A, n(1)),
	matches_replaces_mul_result([t(*,A^N), t(/,A)], [t(*,A^(N-n(1))), t(/,n(1))], T, U).
% Extract sign
f_simple(T, U) :-
	matches_replaces_addition_result([t(+, Z)], [t(-, NewZ)], T, U),
	matches_replaces_mul_result([t(Op,n(N))], [t(Op,n(M))], Z, NewZ),
	N < 0,
	M is N * (-1).
f_simple(T, U) :-
	matches_replaces_addition_result([t(-, Z)], [t(+, NewZ)], T, U),
	matches_replaces_mul_result([t(Op,n(N))], [t(Op,n(M))], Z, NewZ),
	N < 0,
	M is N * (-1).
f_simple(T, U) :-
	Ts = [_,_|_],
	addition_terms(T, Ts),
	maplist(\t(-,A)^t(+,A)^true, Ts, Us),
	terms_addition(Us, U1),
	U = n(-1) * U1.
% ln
f_simple(ln(A^B), B*ln(A)).
f_simple(ln(e), n(1)).
% sin
f_simple(sin(n(0)), n(0)).
f_simple(sin(pi/n(2)), n(1)).
f_simple(sin(n(1)/n(2)*pi), n(1)).
f_simple(sin(pi), n(0)).
f_simple(sin(n(3)/n(2)*pi), n(-1)).
% cos
f_simple(cos(n(0)), n(1)).
f_simple(cos(pi/n(2)), n(0)).
f_simple(cos(n(1)/n(2)*pi), n(0)).
f_simple(cos(pi), n(-1)).
f_simple(cos(n(3)/n(2)*pi), n(0)).
% tan
f_simple(tan(n(0)), n(0)).
f_simple(tan(pi), n(0)).
f_simple(T, U) :-
	t_optpow(Sin, sin(A), P),
	t_optpow(Cos, cos(A), P),
	matches_replaces_mul_result([t(*, Sin), t(/, Cos)], [t(*,tan(A)^P), t(/, n(1))], T, U).
f_simple(T, U) :-
	t_optpow(Tan, tan(A), P),
	matches_replaces_mul_result([t(/, Tan)], [t(*,cot(A)^P)], T, U).
% cot
f_simple(cot(pi/n(2)), n(0)).
f_simple(cot(n(1)/n(2)*pi), n(0)).
f_simple(T, U) :-
	t_optpow(Sin, sin(A), P),
	t_optpow(Cos, cos(A), P),
	matches_replaces_mul_result([t(*, Cos), t(/, Sin)], [t(*,cot(A)^P), t(/, n(1))], T, U).
f_simple(T, U) :-
	t_optpow(Cot, cot(A), P),
	matches_replaces_mul_result([t(/, Cot)], [t(*,tan(A)^P)], T, U).
% Factor common terms (distributive law)
f_simple(T, U) :-
	dif(A, n(0)),
	dif(A, n(1)),
	addition_terms(T, Ts),
	length(Ts, TsLen),
	TsLen > 1,
	maplist({DOp,A}+\t(Op,LT)^t(Op,LU)^matches_replaces_mul_result([t(DOp,A)], [t(DOp,n(1))], LT, LU), Ts, Ds),
	terms_addition(Ds, D),
	binop_type_arg_arg(U, DOp, D, A).
% Sort the terms
f_simple(T, U) :-
	Ts = [_,_|_],
	addition_terms(T, Ts),
	dif(Ts, Us),
	predsort(rel_plus_termorder, Ts, Us),
	terms_addition(Us, U).
f_simple(T, U) :-
	Ts = [_,_|_],
	mul_terms(T, Ts),
	dif(Ts, Us),
	predsort(rel_mul_termorder, Ts, Us),
	terms_mul(Us, U).
% Fallback (recursion)
f_simple(T, U) :-
	binop_type_arg_arg(T, Op, A, B),
	f_simple(A, AS),
	f_simple(B, BS),
	binop_type_arg_arg(U, Op, AS, BS).
f_simple(T, U) :-
	fn_type_arg(T, Fn, A),
	f_simple(A, AS),
	fn_type_arg(U, Fn, AS).
f_simple(C, C) :-
	constant(C).
f_simple(V, V) :-
	variable(V).

f_maxsimple_(F, [F]) :-
	dif(F, S),
	not(f_simple(F, S)),
	!.
f_maxsimple_(F, [S|Ss]) :-
	dif(F, S),
	f_simple(F, S),
	!,
	%print(S), nl,
	term(S),  % security check
	f_maxsimple_(S, Ss).

f_maxsimple(F, S) :-
	f_maxsimple_(F, Ss),
	lastelement_of(S, Ss).

:- f_maxsimple(cos(x)+ln(x)+tan(x)+cot(x)+sin(x), ln(x)+sin(x)+cos(x)+tan(x)+cot(x)).
:- f_maxsimple(cos(x)+cos(x^n(2))+cos(x)^n(2), cos(x^n(2))+cos(x)+cos(x)^n(2)).
:- f_maxsimple(x^n(2) + (n(1)+x)^n(2) + x^n(3), (x+n(1))^n(2) + x^n(3) + x^n(2)).
:- f_maxsimple(n(3)+x^n(4)+x^x, x^x+x^n(4)+n(3)).
:- f_maxsimple(n(5)^x+n(3)+x^n(5), x^n(5)+n(5)^x+n(3)).
:- f_maxsimple(n(3) + (n(2) - x^n(1) + x^n(2))^n(3) + n(2)*x^n(4),
	           (x^n(2) - x + n(2))^n(3) + n(2)*x^n(4) + n(3)).
:- f_maxsimple(x + x*x*x + x*x + x*x*x*x, x* (x* (x* (x+n(1))+n(1))+n(1))).
:- f_maxsimple(x+x+x+x+x+x+x, n(7)*x).
:- f_maxsimple(cos(x)-ln(x), n(-1)*ln(x)+cos(x)).

:- f_maxsimple(x^x*n(4), n(4)*x^x).
:- f_maxsimple((n(1)+n(2)+n(3))*n(5), n(30)).
:- f_maxsimple(x*n(4), n(4)*x).
:- f_maxsimple(x*n(4)*x, n(4)*x^n(2)).

:- f_maxsimple(n(1) + x^n(3) + x^n(5)*n(4), n(4)*x^n(5) + x^n(3) + n(1)).
:- f_maxsimple(n(4)*x^n(4)/n(3) + (x^x)/n(4), x^x/n(4) + n(4)*x^n(4)/n(3)).
:- f_maxsimple(n(4)*x^n(4)/n(-3) + (x^x)/n(-4), n(-1) * (x^x/n(4) + n(4)*x^n(4)/n(3))).
:- f_maxsimple(n(2)/n(4), n(1)/n(2)).
:- f_maxsimple(n(4)/n(2), n(2)).

:- f_maxsimple(x^n(-1), n(1)/x).
:- f_maxsimple((x*x) + n(4) + (x*x), n(2)*x^n(2) + n(4)).
:- f_maxsimple((n(1)+x)^n(2) * (n(3)^x) * (n(1)+x)^n(3), n(3)^x * (x+n(1))^n(5)).

:- f_maxsimple(x*n(1)/x, n(1)).
:- f_maxsimple(x^n(2) * n(1)/x, x).
:- f_maxsimple(x^n(3) * n(1)/x^n(2), x).
:- f_maxsimple(x^n(3) * n(1)/x, x^n(2)).
:- f_maxsimple(x * n(1)/x^n(4), n(1)/x^n(3)).

:- f_maxsimple(x*ln(x)*n(1)/x, ln(x)).
:- f_maxsimple(x^n(2) * n(1)/(x * ln(x)), x/ln(x)).
:- f_maxsimple(x^n(3) * n(1)/(ln(x)*x^n(2)), x/ln(x)).

:- f_maxsimple(n(0)/ln(x), n(0)).
:- f_maxsimple(n(0)+x-n(0)-n(0), x).
:- f_maxsimple(x+x-n(0), n(2)*x).
:- f_maxsimple(n(1)-n(2), n(-1)).

:- f_maxsimple(n(2)*(n(2)*x + n(2)*e^x), n(4)*(e^x + x)).

:- f_maxsimple(n(1)-n(1), n(0)).
:- f_maxsimple(x+ n(-1)*n(4), x - n(4)).
:- f_maxsimple(x^n(2) + n(-1)*x, x^n(2) - x).
:- f_maxsimple(x^n(2) - n(-1)*x, x^n(2) + x).
:- f_maxsimple(x^n(2) - x*n(1)*n(-1), x^n(2) + x).

:- f_maxsimple(x-x^n(2), n(-1)*x^n(2) + x).
:- f_maxsimple(n(1)/cos(x)-n(1)/cos(x), n(0)).
:- f_maxsimple(n(1)/cos(x)/sin(x) - n(1)/cos(x), (n(1)/sin(x) - n(1))/cos(x)).

:- f_maxsimple(n(-1)*ln(x)-cos(x)-cot(x), n(-1)*(ln(x)+cos(x)+cot(x))).

:- f_maxsimple(n(5)/tan(x^n(4))^n(2), n(5)*cot(x^n(4))^n(2)).
:- f_maxsimple(n(5)/cot(x^n(4))^n(2), n(5)*tan(x^n(4))^n(2)).

:- f_ddx(tan(x), D), f_maxsimple(D, tan(x)^n(2)+n(1)).
:- f_ddx(cot(x), D), f_maxsimple(D, n(-1)*(cot(x)^n(2)+n(1))).
:- f_ddx(ln(ln(n(1)/x)), D), f_maxsimple(D, n(-1)/x/ln(n(1)/x)).
:- f_ddx(sin(cos(n(1)/x)), D), f_maxsimple(D, sin(n(1)/x)*cos(cos(n(1)/x))/x^n(2)).
:- f_ddx(cos(x)/cos(x), D), f_maxsimple(D, n(0)).
:- f_ddx(x*e^(pi*x), D), f_maxsimple(D, e^ (pi*x)* (pi*x+n(1))).
:- f_ddx(sin(x^n(2))^n(2), D), f_maxsimple(D, n(4)*x*sin(x^n(2))*cos(x^n(2))).
:- f_ddx(ln(ln(n(1)/x)/x)/x, D), f_maxsimple(D, n(-1)* (ln(ln(n(1)/x)/x)+ (ln(n(1)/x)+n(1))/ln(n(1)/x))/x^n(2)).

% Does cleanup work that interferes with the simplification process
% Chains of / will be merged to mulitplications: a*b/c/d/e = a*b/(c*d*e)
f_postsimple(T, U) :-
	Divs = [_|_],
	Muls = [_|_],
	mul_terms(T, Ts),
	append(Divs, Muls, Ts),
	maplist(\t(/,A)^t(*,B)^f_postsimple(A, B), Divs, DivsAsMuls),
	maplist(\t(*,A)^t(*,B)^f_postsimple(A, B), Muls, MulsOpt),
	!,
	terms_mul(MulsOpt, U1),
	terms_mul(DivsAsMuls, U2),
	U = U1 / U2.
% Recursion
f_postsimple(T, U) :-
	binop_type_arg_arg(T, Op, A, B),
	f_postsimple(A, AS),
	f_postsimple(B, BS),
	binop_type_arg_arg(U, Op, AS, BS).
f_postsimple(T, U) :-
	fn_type_arg(T, Fn, A),
	f_postsimple(A, AS),
	fn_type_arg(U, Fn, AS).
f_postsimple(C, C) :-
	constant(C).
f_postsimple(V, V) :-
	variable(V).

:- f_postsimple(n(1)*n(2)*n(3)/n(4)/n(5)/n(6), n(1)*n(2)*n(3)/(n(4)*n(5)*n(6))).
:- f_postsimple(n(1)/n(2)/n(3) + n(4)/n(5), n(1)/(n(2)*n(3)) + n(4)/n(5)).
