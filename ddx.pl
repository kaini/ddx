
main(Formula, Tex) :-
	f_ddx(Formula, Dx),
	f_maxsimple(Dx, Simple),
	phrase(term_latex_root(Simple), Tex).

main_print(Formula) :-
	main(Formula, Tex),
	format('~s~n', [Tex]).

% Helper libraries % imports
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

constant(e).
constant(pi).
constant(N) :- mynumber(N).

variable(x).

binop_type_arg_arg(A+B, plus, A, B).
binop_type_arg_arg(A-B, minus, A, B).
binop_type_arg_arg(A*B, mul, A, B).
binop_type_arg_arg(A/B, div, A, B).
binop_type_arg_arg(A^B, pow, A, B).

fn_type_arg(ln(A), ln, A).
fn_type_arg(sin(A), sin, A).
fn_type_arg(cos(A), cos, A).
fn_type_arg(tan(A), tan, A).
fn_type_arg(cot(A), cot, A).

term(V) :- variable(V).
term(C) :- constant(C).
term(BO) :- binop_arg_arg(BO, _, A, B), term(A), term(B).
term(FN) :- fn_arg(FN, _, A), term(A).

term_latex_root(T) -->
	"\\int ", term_latex(T, false, true), " \\:\\textup{d}x".

% arguments: (term, multiplication brace, addition brace)
term_latex(V, _, _) -->
	{ variable(V) },
	variable_latex(V).
term_latex(C, _, _) -->
	{ constant(C) },
	constant_latex(C).
term_latex(A+B, _, AddBr) -->
	obrace(AddBr),
	term_latex(A, false, false),
	" + ",
	term_latex(B, false, false),
	cbrace(AddBr).
term_latex(A-B, _, AddBr) -->
	obrace(AddBr),
	term_latex(A, false, false),
	" - ",
	term_latex(B, false, true),
	cbrace(AddBr).
term_latex(A*B, MulBr, _) -->
	obrace(MulBr),
	term_latex(A, false, true),
	" ",
	term_latex(B, false, true),
	cbrace(MulBr).
term_latex(A/B, _, _) -->
	"\\frac{",
	term_latex(A, false, false),
	"}{",
	term_latex(B, false, false),
	"}".
term_latex(A^B, MulBr, _) -->
	{ dif(B, 1/2) },
	obrace(MulBr),
	term_latex(A, true, true),
	"^{",
	term_latex(B, false, false),
	"}",
	cbrace(MulBr).
term_latex(A^(1/2), _, _) -->
	"\\sqrt{",
	term_latex(A, false, false),
	"}".
term_latex(sin(T), _, _) -->
	"\\textup{sin}\\left(", term_latex(T, false, false), "\\right)".
term_latex(cos(T), _, _) -->
	"\\textup{cos}\\left(", term_latex(T, false, false), "\\right)".
term_latex(tan(T), _, _) -->
	"\\textup{tan}\\left(", term_latex(T, false, false), "\\right)".
term_latex(cot(T), _, _) -->
	"\\textup{cot}\\left(", term_latex(T, false, false), "\\right)".
term_latex(ln(T), _, _) -->
	"\\textup{ln}\\left(", term_latex(T, false, false), "\\right)".

obrace(false) -->
	"".
obrace(true) -->
	"\\left(".

cbrace(false) -->
	"".
cbrace(true) -->
	"\\right)".

constant_latex(e) -->
	"e".
constant_latex(pi) -->
	"\\pi".
constant_latex(A rdiv B) -->
	term_latex(A/B, false, false).
constant_latex(N) -->
	{ integer(N) },
	{ N >= 0 },
	{ format(string(Tex), '~d', [N]) },
	Tex.
constant_latex(N) -->
	{ integer(N) },
	{ N < 0 },
	{ format(string(Tex), '(~d)', [N]) },
	Tex.

variable_latex(x) -->
	"x".

% Differentiation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basic rules
f_ddx(V, 1) :-
	variable(V).
f_ddx(C, 0) :-
    constant(C).
% Core rules: addition, multiplication and power
f_ddx(A+B, AA+BB) :-
    f_ddx(A, AA),
    f_ddx(B, BB).
f_ddx(A*B, (AA*B + A*BB)) :-
    f_ddx(A, AA),
    f_ddx(B, BB).
f_ddx(A^B, (A^(B-1)) * (B*AA + A*ln(A)*BB)) :-
    f_ddx(A, AA),
    f_ddx(B, BB).
% Elementary functions
f_ddx(ln(A), A^(-1) * AA) :-
    f_ddx(A, AA).
f_ddx(sin(A), cos(A) * AA) :-
	f_ddx(A, AA).
f_ddx(cos(A), (-1) * sin(A) * AA) :-
	f_ddx(A, AA).
% Extra rules: These terms will be differentiated by changing them to ones that
% can be differentiated.
f_ddx(A-B, D) :-
	f_ddx(A + (-1)*B, D).
f_ddx(A/B, D) :-
	f_ddx(A * B^(-1), D).
f_ddx(tan(A), D) :-
	f_ddx(sin(A) / cos(A), D).
f_ddx(cot(A), D) :-
	f_ddx(cos(A) / sin(A), D).


:- f_ddx(x, 1).
:- f_ddx(123, 0).
:- f_ddx(0, 0).
:- f_ddx(+(10, x), +(0, 1)).
:- f_ddx(+(x, x), +(1, 1)).
:- f_ddx(*(x, x), +(*(1, x), *(x, 1))).
:- f_ddx(ln(x), *(^(x, -1), 1)).
:- f_ddx(ln(*(2, x)), *(^(*(2, x), -1), +(*(0, x), *(2, 1)))). 

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
	% TODO thats HIGHLY ugly but mynumber/1 is unfortunately not relational and so is constant/1 :(
	length(Order, 9),
	before_after_inlist(A, B, Order),
	additionorder(Order).
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
plus_termorder(e, N) :-
	mynumber(N).
plus_termorder(pi, N) :-
	mynumber(N).
plus_termorder(N, M) :-
	mynumber(N),
	mynumber(M),
	N > M.

rel_plus_termorder('>', A, B) :- plus_termorder(A, B), !.
rel_plus_termorder('<', _, _).

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
	% TODO thats HIGHLY ugly but mynumber/1 is unfortunately not relational and so is constant/1 :(
	length(Order, 9),
	before_after_inlist(A, B, Order),
	mulorder(Order).
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
mul_termorder(N, e) :-
	mynumber(N).
mul_termorder(N, pi) :-
	mynumber(N).
mul_termorder(N, M) :-
	mynumber(N),
	mynumber(M),
	N < M.
mul_termorder(e, pi).

rel_mul_termorder('>', A, B) :- mul_termorder(A, B), !.
rel_mul_termorder('<', _, _).

% Collect all terms in a (left assoc.) multiplication chain
mul_terms(A, [A]) :-
	not(A = _*_),
	not(A = _/_).
mul_terms(1/B, [1/B]).
mul_terms(A/B, [1/B|Us]) :-
	dif(A, 1),
	mul_terms(A, Us).
mul_terms(A*B, [B|Us]) :-
	mul_terms(A, Us).

terms_mul_([T], T).
terms_mul_([1/T|Ts], R/T) :-
	terms_mul_(Ts, R).
terms_mul_([T|Ts], R*T) :-
	terms_mul_(Ts, R).
terms_mul([], 1).
terms_mul(Xs, Mul) :-
	terms_mul_(Xs, Mul).

matches_replaces_mul_result(Ms, Rs, T, U) :-
	mul_terms(T, Ts),
	matches_replaces_list_list(Ms, Rs, Ts, Us),
	terms_mul(Us, U).

:- matches_replaces_mul_result([A,A], [A^2,1], 1*2*1, 1^2*2*1).

% Collect all terms in a (left associative) addition chain
addition_terms(A, [t(plus, A)]) :-
	not(A = _+_),
	not(A = _-_).
addition_terms(A-B, [t(minus, B)|Us]) :-
	addition_terms(A, Us).
addition_terms(A+B, [t(plus, B)|Us]) :-
	addition_terms(A, Us).

terms_addition_([t(plus, T)], T).
terms_addition_([t(minus, (-1)*T)], (-1)*T).
terms_addition_([t(minus, T)], (-1)*T) :-
	not(T = (-1)*_).
terms_addition_([t(minus, T)|Ts], R-T) :-
	terms_addition_(Ts, R).
terms_addition_([t(plus, T)|Ts], R+T) :-
	terms_addition_(Ts, R).
terms_addition([], 0).
terms_addition(Xs, Add) :-
	terms_addition_(Xs, Add).

matches_replaces_addition_result(Ms, Rs, T, U) :-
	addition_terms(T, Ts),
	matches_replaces_list_list(Ms, Rs, Ts, Us),
	terms_addition(Us, U).

:- matches_replaces_addition_result([t(plus,A), t(plus,A)], [t(plus,2*A),t(plus,0)], 1+2+1, 2*1+2+0).

% Simplification rules, unfortunately the order is relevant, because later rules
% might invoke a endless loop especially if the + or * associativity is not fixed
% yet. :(
% Fix associativity
f_simple(A+(B+C), A+B+C).
f_simple(A*(B*C), A*B*C).
% Addition & Subdraction
f_simple(T, U) :-
	addition_terms(T, Ts),
	member(t(_,0), Ts),
	filter_list_list(t(_,0), Ts, Us),
	terms_addition(Us, U).
f_simple(T, U) :-
	matches_replaces_addition_result([t(plus,A), t(plus,B)], [t(plus,N), t(plus,0)], T, U),
	mynumber(A),
	mynumber(B),
	N is A + B.
f_simple(T, U) :-
	matches_replaces_addition_result([t(plus,A), t(minus,B)], [t(plus,N), t(minus,0)], T, U),
	mynumber(A),
	mynumber(B),
	N is A - B.
f_simple(T, U) :-
	dif(A, 0),
	matches_replaces_addition_result([t(Op,A), t(Op,A)], [t(Op,2*A), t(Op,0)], T, U).
f_simple(T, U) :-
	dif(A, 0),
	matches_replaces_addition_result([t(Op,N*A), t(Op,A)], [t(Op,(N+1)*A), t(Op,0)], T, U),
	mynumber(N).
f_simple(T, U) :-
	dif(A, 0),
	matches_replaces_addition_result([t(Op,A*N), t(Op,A)], [t(Op,(N+1)*A), t(Op,0)], T, U),
	mynumber(N).
% Multiplication
f_simple(T, 0) :-
	mul_terms(T, Ts),
	member(0, Ts).
f_simple(T, U) :-
	mul_terms(T, Ts),
	member(1, Ts),
	filter_list_list(1, Ts, Us),
	terms_mul(Us, U).
f_simple(T, U) :-
	matches_replaces_mul_result([A, B], [N, 1], T, U),
	mynumber(A),
	mynumber(B),
	N is A * B.
f_simple(T, U) :-
	dif(A, 1),
	dif(A, 0),
	matches_replaces_mul_result([A, A], [A^2, 1], T, U).
f_simple(T, U) :-
	dif(A, 1),
	dif(A, 0),
	matches_replaces_mul_result([A, 1/A], [1, 1], T, U).
% Division
f_simple(T, U) :-
	matches_replaces_mul_result([A/B], [N], T, U),
	mynumber(A),
	mynumber(B),
	N is A rdiv B.
f_simple(T, U) :-
	matches_replaces_mul_result([A, 1/B], [N, 1], T, U),
	mynumber(A),
	mynumber(B),
	N is A rdiv B.
f_simple(T, U) :-
	mul_terms(T, Ts),
	member(1/1, Ts),
	filter_list_list(1/1, Ts, Us),
	terms_mul(Us, U).

f_simple(A/(B/C), (A*C)/B).
f_simple((A/B)/C, A/(B*C)).
f_simple(T, U) :-
	matches_replaces_mul_result([A/B, C/D], [(A*C)/(B*D), 1], T, U).
% Power
f_simple(_^0, 1).
f_simple(A^1, A).
f_simple(1^_, 1).
f_simple(0^_, 0).
f_simple(A^B, N) :-
	mynumber(A),
	mynumber(B),
	N is A ^ B.
f_simple(A^N, 1/(A^M)) :-
	mynumber(N),
	N < 0,
	M is (-1) * N.
f_simple((A^B)^C, A^(B*C)).
f_simple(T, U) :-
	matches_replaces_mul_result([A^N, A], [A^(N+1), 1], T, U).
f_simple(T, U) :-
	matches_replaces_mul_result([A^B, A^C], [A^(B+C), 1], T, U).
% Kürzi kürzi
f_simple(T, U) :-
	dif(A, 0),
	dif(A, 1),
	matches_replaces_mul_result([A^N, 1/Z], [A^(N-M), 1/NewZ], T, U),
	matches_replaces_mul_result([A^M], [1], Z, NewZ).
f_simple(T, U) :-
	dif(A, 0),
	dif(A, 1),
	matches_replaces_mul_result([A, 1/Z], [1, 1/NewZ], T, U),
	matches_replaces_mul_result([A^N], [A^(N-1)], Z, NewZ).
f_simple(T, U) :-
	dif(A, 0),
	dif(A, 1),
	matches_replaces_mul_result([A^N, 1/Z], [A^(N-1), 1/NewZ], T, U),
	matches_replaces_mul_result([A], [1], Z, NewZ).
% Extract sign
f_simple(T, U) :-
	matches_replaces_addition_result([t(plus, Z)], [t(minus, NewZ)], T, U),
	matches_replaces_mul_result([N], [M], Z, NewZ),
	mynumber(N),
	N < 0,
	M is N * (-1).
f_simple(T, U) :-
	matches_replaces_addition_result([t(minus, Z)], [t(plus, NewZ)], T, U),
	matches_replaces_mul_result([N], [M], Z, NewZ),
	mynumber(N),
	N < 0,
	M is N * (-1).
% ln
f_simple(ln(A^B), B*ln(A)).
f_simple(ln(e), 1).
% sin
f_simple(sin(0), 0).
f_simple(sin(pi/2), 1).
f_simple(sin(1/2*pi), 1).
f_simple(sin(pi), 0).
f_simple(sin(3/2*pi), -1).
% cos
f_simple(cos(0), 1).
f_simple(cos(pi/2), 0).
f_simple(cos(1/2*pi), 0).
f_simple(cos(pi), -1).
f_simple(cos(3/2*pi), 0).
% tan
f_simple(tan(0), 0).
f_simple(tan(pi), 0).
% cot
f_simple(cot(pi/2), 0).
f_simple(cot(1/2*pi), 0).
% Fallback (recursion)
f_simple(A+B, AS+BS) :- f_simple(A, AS), f_simple(B, BS).
f_simple(A-B, AS-BS) :- f_simple(A, AS), f_simple(B, BS).
f_simple(A*B, AS*BS) :- f_simple(A, AS), f_simple(B, BS).
f_simple(A/B, AS/BS) :- f_simple(A, AS), f_simple(B, BS).
f_simple(A^B, AS^BS) :- f_simple(A, AS), f_simple(B, BS).
f_simple(ln(A), ln(AS)) :- f_simple(A, AS).
f_simple(cos(A), cos(AS)) :- f_simple(A, AS).
f_simple(sin(A), sin(AS)) :- f_simple(A, AS).
f_simple(tan(A), tan(AS)) :- f_simple(A, AS).
f_simple(cot(A), cot(AS)) :- f_simple(A, AS).
% Raushebi raushebi
f_simple(T, U) :-
	dif(A, 0),
	dif(A, 1),
	addition_terms(T, Ts),
	length(Ts, TsLen),
	TsLen > 1,
	maplist({A}+\t(Op,LT)^t(Op,LU)^matches_replaces_mul_result([A], [1], LT, LU), Ts, Ds),
	terms_addition(Ds, D),
	U = A*D.
% Sort the terms
f_simple(T, U) :-
	addition_terms(T, Ts),
	dif(Ts, Us),
	predsort(\R^t(_,A)^t(_,B)^rel_plus_termorder(R,A,B), Ts, Us),
	terms_addition(Us, U).
f_simple(T, U) :-
	mul_terms(T, Ts),
	dif(Ts, Us),
	predsort(rel_mul_termorder, Ts, Us),
	terms_mul(Us, U).
% Fallback (nothing else matches)
f_simple(F, F).

f_maxsimple_(F, [F]) :-
	dif(F, S),
	not(f_simple(F, S)).
	%nl, nl.
f_maxsimple_(F, [S|Ss]) :-
	dif(F, S),
	f_simple(F, S),
	!,
	print(S), nl,
	f_maxsimple_(S, Ss).

f_maxsimple(F, S) :-
	f_maxsimple_(F, Ss),
	lastelement_of(S, Ss).

:- f_maxsimple(a+b+c, a+b+c).
:- f_maxsimple(a+(b+c), a+b+c).
:- f_maxsimple((a+b)+(c+d), a+b+c+d).
:- f_maxsimple(a+(b+c)+d, a+b+c+d).
:- f_maxsimple((a+b)+(c+d)+e, a+b+c+d+e).
:- f_maxsimple(a+(b+(c+(d+e))), a+b+c+d+e).
:- f_maxsimple((a*b)*(c*d)*e, a*b*c*d*e).
:- f_maxsimple(a*(b*(c*d)), a*b*c*d).

:- f_maxsimple(cos(x)+ln(x)+tan(x)+cot(x)+sin(x), ln(x)+sin(x)+cos(x)+tan(x)+cot(x)).
:- f_maxsimple(cos(x)+cos(x^2)+cos(x)^2, cos(x^2)+cos(x)+cos(x)^2).
:- f_maxsimple(x^2 + (1+x)^2 + x^3, (x+1)^2 + x^3 + x^2).
:- f_maxsimple(3+x^4+x^x, x^x+x^4+3).
:- f_maxsimple(5^x+3+x^5, x^5+5^x+3).
:- f_maxsimple(3 + (2 - x^1 + x^2)^3 + 2*x^4, (x^2 - x + 2)^3 + 2*x^4 + 3).
:- f_maxsimple(x + x*x*x + x*x + x*x*x*x, x^4 + x^3 + x^2 + x).
:- f_maxsimple(x+x+x+x+x+x+x, 7*x).

:- f_maxsimple(x^x*4, 4*x^x).
:- f_maxsimple((1+2+3)*5, 30).
:- f_maxsimple(x*4, 4*x).
:- f_maxsimple(x*4*x, 4*x^2).

:- f_maxsimple(1 + x^3 + x^5*4, 4*x^5 + x^3 + 1).
:- f_maxsimple(4*x^4/3 + (x^x)/4, (1 rdiv 4)*x^x + (4 rdiv 3)*x^4).
:- f_maxsimple(4*x^4/(-3) + (x^x)/(-4), (-1 rdiv 4)*x^x - (4 rdiv 3)*x^4).

:- f_maxsimple(x^(-1), 1/x).
:- f_maxsimple((x*x) + 4 + (x*x), 2*x^2 + 4).
:- f_maxsimple((1+x)^2 * (3^x) * (1+x)^3, 3^x * (x+1)^5).

:- f_maxsimple(x*1/x, 1).
:- f_maxsimple(x^2 * 1/x, x).
:- f_maxsimple(x^3 * 1/x^2, x).
:- f_maxsimple(x^3 * 1/x, x^2).
:- f_maxsimple(x * 1/x^4, 1/x^3).

:- f_maxsimple(x*ln(x)*1/x, ln(x)).
:- f_maxsimple(x^2 * 1/(x * ln(x)), x/ln(x)).
:- f_maxsimple(x^3 * 1/(ln(x)*x^2), x/ln(x)).

:- f_maxsimple(0/ln(x), 0).
:- f_maxsimple(0+x-0-0, x).
:- f_maxsimple(x+x-0, 2*x).
:- f_maxsimple(1-2, -1).

:- f_maxsimple(2*(2*x + 2*e^x), 4*(e^x + x)).

:- f_maxsimple(1-1, 0).
:- f_maxsimple(x+ (-1)*4, x - 4).
:- f_maxsimple(x^2 + (-1)*x, x^2 - x).
:- f_maxsimple(x^2 - (-1)*x, x^2 + x).
:- f_maxsimple(x^2 - x*1*(-1), x^2 + x).

:- f_maxsimple(x-x^2, (-1)*x^2 + x).

:- f_ddx(ln(ln(1/x)), D), f_maxsimple(D, -1/(x*ln(1/x))).
:- f_ddx(sin(cos(1/x)), D), f_maxsimple(D, 1/x^2*sin(1/x)*cos(cos(1/x))).
:- f_ddx(cos(x)/cos(x), D), f_maxsimple(D, 0).
:- f_ddx(x*e^(pi*x), D), f_maxsimple(D, e^ (pi*x)* (pi*x+1)).
:- f_ddx(sin(x^2)^2, D), f_maxsimple(D, 4*x*sin(x^2)*cos(x^2)).
