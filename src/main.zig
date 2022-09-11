const std = @import("std");
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;
const GeneralPurposeAllocator = std.heap.GeneralPurposeAllocator;

var gpa = GeneralPurposeAllocator(.{}){};
var arena: ArenaAllocator = ArenaAllocator.init(gpa.allocator());
// var arena: ArenaAllocator = undefined;

extern fn consoleLogString(p: [*]const u8, l: usize) void;

fn logString(string: []const u8) void {
    consoleLogString(string.ptr, string.len);
}

pub fn main() anyerror!void {
    arena = ArenaAllocator.init(gpa.allocator());
}

pub fn log(
    comptime message_level: std.log.Level,
    comptime scope: @Type(.EnumLiteral),
    comptime format: []const u8,
    args: anytype,
) void {
    _ = message_level;
    _ = scope;
    var buffer: [200]u8 = undefined;
    const string = std.fmt.bufPrint(buffer[0..], format, args) catch std.process.exit(2);
    logString(string);
}

pub const os = struct {
    pub const system = struct {
        pub extern fn exit(status: u8) noreturn;
    };
};

fn throwError() noreturn {
    std.process.exit(1);
}

fn signalError(message: []const u8, what_: ?[]const u8) noreturn {
    if (what_) |what| {
        std.log.err("error: {s}: {s}\n", .{ message, what });
    } else {
        std.log.err("error: {s}\n", .{message});
    }
    throwError();
}

const Expr = struct {
    const Kind = enum {
        Func,
        LetRec,
        Var,
        Prim,
        Literal,
        Call,
        If,
    };

    kind: Kind,

    fn init(kind: Kind) Expr {
        return .{ .kind = kind };
    }
};

const Func = struct {
    const arg_count: u32 = 1;

    expr: Expr, // base class
    body: *const Expr, // unique_ptr
    jit_code: ?*anyopaque, // void *

    fn offsetOfBody() usize {
        return @offsetOf(Func, "body");
    }

    fn offsetOfJitCode() usize {
        return @offsetOf(Func, "jit_code");
    }

    fn init(body: *const Expr) Func {
        return .{
            .expr = Expr.init(Expr.Kind.Func),
            .body = body,
            .jit_code = null,
        };
    }
};

const LetRec = struct {
    const arg_count: u32 = 1;

    expr: Expr, // base class
    arg: *const Expr, // unique_ptr
    body: *const Expr, // unique_ptr

    fn init(arg: *const Expr, body: *const Expr) LetRec {
        return .{
            .expr = Expr.init(Expr.Kind.LetRec),
            .arg = arg,
            .body = body,
        };
    }
};

const Var = struct {
    expr: Expr, // base class
    depth: u32,

    fn init(depth: u32) Var {
        return .{
            .expr = Expr.init(Expr.Kind.Var),
            .depth = depth,
        };
    }
};

const Prim = struct {
    const Op = enum {
        Eq,
        LessThan,
        Sub,
        Add,
        Mul,
    };

    expr: Expr, // base class
    op: Op,
    lhs: *const Expr, // unique_ptr
    rhs: *const Expr, // unique_ptr

    fn init(op: Op, lhs: *const Expr, rhs: *const Expr) Prim {
        return .{
            .expr = Expr.init(Expr.Kind.Prim),
            .op = op,
            .lhs = lhs,
            .rhs = rhs,
        };
    }
};

const Literal = struct {
    expr: Expr, // base class
    val: i32,

    fn init(val: i32) Literal {
        return .{
            .expr = Expr.init(Expr.Kind.Literal),
            .val = val,
        };
    }
};

const Call = struct {
    expr: Expr, // base class
    func: *const Expr, // unique_ptr
    arg: *const Expr, // unique_ptr

    fn init(func: *const Expr, arg: *const Expr) Call {
        return .{
            .expr = Expr.init(Expr.Kind.Call),
            .func = func,
            .arg = arg,
        };
    }
};

const If = struct {
    expr: Expr, // base class
    test_: *const Expr, // unique_ptr
    consequent: *const Expr, // unique_ptr
    alternate: *const Expr, // unique_ptr

    fn init(test_: *const Expr, consequent: *const Expr, alternate: *const Expr) If {
        return .{
            .expr = Expr.init(Expr.Kind.If),
            .test_ = test_,
            .consequent = consequent,
            .alternate = alternate,
        };
    }
};

const Parser = struct {
    bound_vars: std.ArrayList([]const u8),
    buf: [*:0]const u8,
    pos: usize,
    len: usize,

    fn isAlphabetic(c: u8) bool {
        return ('a' <= c and c <= 'z') or ('A' <= c and c <= 'Z');
    }

    fn isNumeric(c: u8) bool {
        return '0' <= c and c <= '9';
    }

    fn isAlphaNumeric(c: u8) bool {
        return isAlphabetic(c) or isNumeric(c);
    }

    fn isWhitespace(c: u8) bool {
        return c == ' ' or c == '\t' or c == '\n' or c == '\r';
    }

    fn init(allocator: Allocator, buf: [*:0]const u8) Parser {
        return .{
            .bound_vars = std.ArrayList([]const u8).init(allocator),
            .buf = buf,
            .pos = 0,
            .len = std.mem.indexOfSentinel(u8, 0, buf),
        };
    }

    fn parse(self: *Parser) ?*const Expr {
        const e = self.parseOne();
        self.skipWhitespace();
        if (!self.eof())
            self.err("expected end of input after expression");
        return e;
    }

    fn pushBound(self: *Parser, id: []const u8) void {
        self.bound_vars.append(id) catch std.process.exit(1);
    }

    fn popBound(self: *Parser) void {
        _ = self.bound_vars.pop();
    }

    fn lookupBound(self: Parser, id: []const u8) u32 {
        var i: usize = 0;
        while (i < self.bound_vars.items.len) : (i += 1) {
            if (std.mem.eql(u8, self.bound_vars.items[self.bound_vars.items.len - i - 1], id))
                return i;
        }
        signalError("unbound identifier", id);
        return @bitCast(u32, @as(i32, -1));
    }

    fn err(self: Parser, message: []const u8) noreturn {
        signalError(message, if (self.eof()) self.buf[0..self.len] else self.buf[self.pos..self.len]);
    }

    fn eof(self: Parser) bool {
        return self.pos == self.len;
    }

    fn peek(self: Parser) u8 {
        if (self.eof()) return 0;
        return self.buf[self.pos];
    }

    fn advance(self: *Parser) void {
        if (!self.eof()) self.pos += 1;
    }

    fn next(self: *Parser) u8 {
        const ret = self.peek();
        self.advance();
        return ret;
    }

    fn matchChar(self: *Parser, c: u8) bool {
        if (self.eof() or self.peek() != c)
            return false;
        self.advance();
        return true;
    }

    fn skipWhitespace(self: *Parser) void {
        while (!self.eof() and isWhitespace(self.peek()))
            self.advance();
    }

    fn startsIdentifier(self: Parser) bool {
        return !self.eof() and isAlphabetic(self.peek());
    }

    fn continuesIdentifier(self: Parser) bool {
        return !self.eof() and isAlphaNumeric(self.peek());
    }

    fn matchIdentifier(self: *Parser, literal: []const u8) bool {
        const match_len = literal.len;
        if (match_len < (self.len - self.pos))
            return false;
        // if (strncmp(self.buf + self.pos, literal, match_len) != 0)
        if (!std.mem.eql(u8, self.buf[self.pos .. self.pos + match_len], literal))
            return false;
        if ((self.len - self.pos) < match_len and isAlphaNumeric(self.buf[self.pos + match_len]))
            return false;
        self.pos += match_len;
        return true;
    }

    fn takeIdentifier(self: *Parser) []const u8 {
        const start = self.pos;
        while (self.continuesIdentifier())
            self.advance();
        const end = self.pos;
        return self.buf[start..end];
    }

    fn matchKeyword(self: *Parser, kw: []const u8) bool {
        const kwlen = kw.len;
        var remaining = self.len - self.pos;
        if (remaining < kwlen)
            return false;
        if (!std.mem.eql(u8, self.buf[self.pos .. self.pos + kwlen], kw))
            return false;
        self.pos += kwlen;
        if (!self.continuesIdentifier())
            return true;
        self.pos -= kwlen;
        return false;
        // if ((self.len - self.pos) < kwlen and isalnum(self.buf[self.pos + kwlen]))
        //   return 0;
        // self.pos += kwlen;
        // return 1;
    }

    fn parsePrim(self: *Parser, op: Prim.Op) *const Expr {
        const lhs = self.parseOne();
        const rhs = self.parseOne();
        return &Prim.init(op, lhs.?, rhs.?).expr;
    }

    fn parseInt32(self: *Parser) i32 {
        var ret: u64 = 0;
        while (!self.eof() and isNumeric(self.peek())) {
            ret *= 10;
            ret += self.next() - '0';
            if (ret > 0x7fffffff)
                self.err("integer too long");
        }
        if (!self.eof() and !isWhitespace(self.peek()) and self.peek() != ')')
            self.err("unexpected integer suffix");
        return @intCast(i32, ret);
    }

    fn parseOne(self: *Parser) ?*const Expr {
        self.skipWhitespace();
        if (self.eof())
            self.err("unexpected end of input");
        if (self.matchChar('(')) {
            self.skipWhitespace();

            var ret: *const Expr = undefined;
            if (self.matchKeyword("lambda")) {
                self.skipWhitespace();
                if (!self.matchChar('('))
                    self.err("expected open paren after lambda");
                self.skipWhitespace();
                if (!self.startsIdentifier())
                    self.err("expected an argument for lambda");
                self.pushBound(self.takeIdentifier());
                self.skipWhitespace();
                if (!self.matchChar(')'))
                    self.err("expected just one argument for lambda");
                const body = self.parseOne();
                self.popBound();
                ret = &Func.init(body.?).expr;
            } else if (self.matchKeyword("letrec")) {
                self.skipWhitespace();
                if (!self.matchChar('('))
                    self.err("expected open paren after letrec");
                if (!self.matchChar('('))
                    self.err("expected two open parens after letrec");
                self.skipWhitespace();
                if (!self.startsIdentifier())
                    self.err("expected an identifier for letrec");
                self.pushBound(self.takeIdentifier());
                self.skipWhitespace();
                const arg = self.parseOne();
                if (!self.matchChar(')'))
                    self.err("expected close paren after letrec binding");
                self.skipWhitespace();
                if (!self.matchChar(')'))
                    self.err("expected just one binding for letrec");
                const body = self.parseOne();
                self.popBound();
                ret = &LetRec.init(arg.?, body.?).expr;
            } else if (self.matchKeyword("+")) {
                ret = self.parsePrim(Prim.Op.Add);
            } else if (self.matchKeyword("-")) {
                ret = self.parsePrim(Prim.Op.Sub);
            } else if (self.matchKeyword("<")) {
                ret = self.parsePrim(Prim.Op.LessThan);
            } else if (self.matchKeyword("eq?")) {
                ret = self.parsePrim(Prim.Op.Eq);
            } else if (self.matchKeyword("*")) {
                ret = self.parsePrim(Prim.Op.Mul);
            } else if (self.matchKeyword("if")) {
                const test_ = self.parseOne();
                const consequent = self.parseOne();
                const alternate = self.parseOne();
                ret = &If.init(test_.?, consequent.?, alternate.?).expr;
            } else {
                // Otherwise it's a call.
                const func = self.parseOne();
                const arg = self.parseOne();
                ret = &Call.init(func.?, arg.?).expr;
            }
            self.skipWhitespace();
            if (!self.matchChar(')'))
                self.err("expected close parenthesis");
            return ret;
        } else if (self.startsIdentifier()) {
            return &Var.init(self.lookupBound(self.takeIdentifier())).expr;
        } else if (isNumeric(self.peek())) {
            return &Literal.init(self.parseInt32()).expr;
        }
        self.err("unexpected input");
        return null;
    }
};

export fn parse(str: [*:0]const u8) ?*const Expr {
    std.log.info("parser input: {s}", .{str});
    var parser = Parser.init(arena.allocator(), str);
    const result = parser.parse();
    std.log.info("parser output: {}", .{result.?});
    return result;
}

const HeapObject = struct {
    const Kind = enum(usize) {
        Env,
        Closure,
    };

    const NotForwardedBit: usize = 1;
    const NotForwardedBits: usize = 1;
    const NotForwardedBitMask: usize = (1 << NotForwardedBits) - 1;

    fn offsetOfTag() usize {
        return @offsetOf(HeapObject, "tag");
    }

    tag: usize,

    // fn init(kind: Kind) HeapObject {
    //     return .{
    //         .tag = (@enumToInt(kind) << NotForwardedBits) | NotForwardedBit,
    //     };
    // }

    fn init(heap: *Heap, bytes: usize, kind_: Kind) HeapObject {
        var self = heap.allocate(bytes);
        self.* = .{
            .tag = (@enumToInt(kind_) << NotForwardedBits) | NotForwardedBit,
        };
        return self.*;
    }

    fn isForwarded(self: HeapObject) bool {
        return (self.tag & NotForwardedBit) == 0;
    }

    fn forwarded(self: HeapObject) *HeapObject {
        return @intToPtr(*HeapObject, self.tag);
    }

    fn forward(self: *HeapObject, new_loc: *HeapObject) void {
        self.tag = @ptrToInt(new_loc);
    }

    fn kind(self: HeapObject) Kind {
        return @intToEnum(Kind, self.tag >> 1);
    }

    fn isEnv(self: HeapObject) bool {
        return self.kind() == .Env;
    }

    fn isClosure(self: HeapObject) bool {
        return self.kind() == .Closure;
    }

    fn asEnv(self: *HeapObject) *Env {
        return @fieldParentPtr(Env, "obj", self);
    }

    fn asClosure(self: *HeapObject) *Closure {
        return @fieldParentPtr(Closure, "obj", self);
    }

    fn kindName(self: HeapObject) []const u8 {
        return @tagName(self.kind());
    }
};

const Heap = struct {
    hp: usize,
    limit: usize,
    base: usize,
    size: usize,
    count: i32,
    mem: []u8,
    roots: std.ArrayList(Value),

    const ALIGNMENT: usize = 8;

    fn alignUp(val: usize) usize {
        return (val + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }

    fn pushRoot(heap: *Heap, v: Value) usize {
        const ret = heap.roots.items.len;
        heap.roots.append(v) catch std.process.exit(1);
        return ret;
    }

    fn getRoot(heap: *Heap, idx: usize) Value {
        return heap.roots.items[idx];
    }

    fn setRoot(heap: *Heap, idx: usize, v: Value) void {
        heap.roots.items[idx] = v;
    }

    fn popRoot(heap: *Heap) void {
        _ = heap.roots.pop();
    }

    fn init(heap_size: usize) Heap {
        const allocator = gpa.allocator();
        const mem = allocator.alloc(u8, alignUp(heap_size)) catch signalError("malloc failed", null);
        const base = @ptrToInt(mem.ptr);
        var self = Heap{
            .hp = base,
            .limit = undefined,
            .base = base,
            .size = heap_size,
            .count = -1,
            .mem = mem,
            .roots = std.ArrayList(Value).init(allocator),
        };
        self.flip();
        return self;
    }

    fn deinit(self: *Heap) void {
        const allocator = gpa.allocator();
        allocator.free(self.mem);
        self.roots.deinit();
    }

    fn flip(self: *Heap) void {
        const split = self.base + (self.size >> 1);
        if (self.hp <= split) {
            self.hp = split;
            self.limit = self.base + self.size;
        } else {
            self.hp = self.base;
            self.limit = split;
        }
        self.count += 1;
    }

    fn copy(self: *Heap, obj: *HeapObject) *HeapObject {
        const size = switch (obj.kind()) {
            .Env => obj.asEnv().byteSize(),
            .Closure => obj.asClosure().byteSize(),
        };
        const new_obj = @intToPtr(*HeapObject, self.hp);
        std.mem.copy(u8, @ptrCast([*]u8, new_obj)[0..size], @ptrCast([*]u8, obj)[0..size]);
        obj.forward(new_obj);
        self.hp += alignUp(size);
        return new_obj;
    }

    fn scan(self: *Heap, obj: *HeapObject) usize {
        switch (obj.kind()) {
            .Env => {
                obj.asEnv().visitFields(self);
                return obj.asEnv().byteSize();
            },
            .Closure => {
                obj.asClosure().visitFields(self);
                return obj.asClosure().byteSize();
            },
            // else => abort(),
        }
    }

    fn visitRoots(self: Heap) void {
        for (self.roots.items) |root| {
            root.visitFields(self);
        }
    }

    fn collect(self: *Heap) void {
        self.flip();
        var grey = self.hp;
        while (grey < self.hp)
            grey += alignUp(self.scan(@intToPtr(*HeapObject, grey)));
    }

    fn visit(self: *Heap, comptime T: type, loc: *?*T) void {
        const obj_ = loc.*;
        if (obj_) |obj| {
            if (T == HeapObject) {
                loc.* = if (obj.isForwarded()) obj.forwarded() else self.copy(obj);
            } else {
                loc.* = @fieldParentPtr(T, "obj", if (obj.obj.isForwarded()) obj.obj.forwarded() else self.copy(&obj.obj));
            }
        }
    }

    fn allocate(self: *Heap, size: usize) *HeapObject {
        while (true) {
            const addr = self.hp;
            const new_hp = alignUp(addr + size);
            if (self.limit < new_hp) {
                self.collect();
                if (self.limit - self.hp < size)
                    signalError("ran out of space", null);
                continue;
            }
            self.hp = new_hp;
            return @intToPtr(*HeapObject, addr);
        }
    }
};

const Value = struct {
    payload: usize,

    const HeapObjectTag: usize = 0;
    const SmiTag: usize = 1;
    const TagBits: usize = 1;
    const TagMask: usize = (1 << TagBits) - 1;

    fn init(obj: ?*HeapObject) Value {
        return .{
            .payload = @ptrToInt(obj),
        };
    }

    fn initVal(val: isize) Value {
        return .{
            .payload = (@intCast(usize, val) << TagBits) | SmiTag,
        };
    }

    fn initBool(b: bool) Value {
        return initVal(@boolToInt(b));
    }

    fn isSmi(self: Value) bool {
        return self.payload & TagBits == SmiTag;
    }

    fn isHeapObject(self: Value) bool {
        return self.payload & TagMask == HeapObjectTag;
    }

    fn getSmi(self: Value) isize {
        return @intCast(isize, self.payload) >> TagBits;
    }

    fn getHeapObject(self: Value) ?*HeapObject {
        return @intToPtr(?*HeapObject, self.payload & ~HeapObjectTag);
    }

    fn bits(self: Value) usize {
        return self.payload;
    }

    fn kindName(self: Value) []const u8 {
        return if (self.isSmi()) "small integer" else self.getHeapObject().?.kindName();
    }

    fn isEnv(self: Value) bool {
        return self.isHeapObject() and self.getHeapObject().?.isEnv();
    }

    fn isClosure(self: Value) bool {
        return self.isHeapObject() and self.getHeapObject().?.isClosure();
    }

    fn asEnv(self: *Value) *Env {
        return self.getHeapObject().asEnv();
    }

    fn asClosure(self: *Value) *Closure {
        return self.getHeapObject().?.asClosure();
    }

    fn visitFields(self: Value, heap: *Heap) void {
        if (self.isHeapObject())
            heap.visit(HeapObject, @ptrCast(*?*HeapObject, &self.payload));
    }
};

fn Rooted(comptime T: type) type {
    return struct {
        heap: *Heap,
        idx: usize,

        const Self = @This();

        fn init(heap: *Heap, obj: if (T == Value) Value else ?*T) Self {

            // fn init(heap: *Heap, obj: *T) Self {
            return .{
                // TODO: check this
                .heap = heap,
                // .heap = Heap.init(heap),
                .idx = Heap.pushRoot(heap, if (T == Value) obj else Value.init(if (obj == null) null else &obj.?.obj)),
            };
        }

        fn deinit(self: Self) void {
            Heap.popRoot(self.heap);
        }

        fn get(self: Self) if (T == Value) Value else ?*T {
            const root = Heap.getRoot(self.heap, self.idx);
            return if (T == Value) root else {
                const obj = root.getHeapObject();
                return if (obj == null) null else @fieldParentPtr(T, "obj", obj.?);
            };
        }

        fn set(self: *Self, obj: if (T == Value) Value else *T) void {
            Heap.setRoot(self.heap, self.idx, if (T == Value) obj else Value.init(&obj.obj));
        }
    };
}

const Env = struct {
    obj: HeapObject,
    prev: *Env,
    val: Value,

    fn offsetOfPrev() usize {
        return @offsetOf(Env, "prev");
    }

    fn offsetOfVal() usize {
        return @offsetOf(Env, "val");
    }

    fn lookup(env: *Env, depth: u32) Value {
        var depth_ = depth;
        var env_ = env;
        while (depth_ > 0) {
            depth_ -= 1;
            env_ = env_.prev;
        }
        return env_.val;
    }

    fn init(heap: *Heap, prev: *Rooted(Env), val: *Rooted(Value)) Env {
        return .{
            .obj = HeapObject.init(heap, @sizeOf(Env), .Env),
            .prev = prev.get().?,
            .val = val.get(),
        };
    }

    fn byteSize(self: Env) usize {
        return @sizeOf(@TypeOf(self));
    }

    fn visitFields(self: *Env, heap: *Heap) void {
        heap.visit(Env, @ptrCast(*?*Env, &self.prev));
        self.val.visitFields(heap);
    }
};

const Closure = struct {
    obj: HeapObject,
    env: ?*Env,
    func: *const Func,

    fn offsetOfEnv() usize {
        return @offsetOf(Closure, "env");
    }

    fn offsetOfFunc() usize {
        return @offsetOf(Closure, "func");
    }

    fn init(heap: *Heap, env: *Rooted(Env), func: *const Func) Closure {
        return .{
            .obj = HeapObject.init(heap, @sizeOf(Closure), .Closure),
            .env = env.get(),
            .func = func,
        };
    }

    fn byteSize(self: Closure) usize {
        return @sizeOf(@TypeOf(self));
    }

    fn visitFields(self: *Closure, heap: *Heap) void {
        heap.visit(Env, &self.env);
    }
};

fn evalPrimcall(op: Prim.Op, lhs: isize, rhs: isize) Value {
    switch (op) {
        .Eq => return Value.initBool(lhs == rhs),
        .LessThan => return Value.initBool(lhs < rhs),
        .Add => return Value.initVal(lhs + rhs),
        .Sub => return Value.initVal(lhs - rhs),
        .Mul => return Value.initVal(lhs * rhs),
        // else => {
        //     signalError("unexpected primcall op", null);
        //     return Value.init(null);
        // },
    }
}

// var jit_candidates: std.AutoHashMap(*Func, void);
var jit_candidates = std.AutoHashMap(*const Func, void).init(gpa.allocator());

const JitFunction = fn (*Env, *Heap) Value;

fn eval_(expr_arg: *const Expr, unrooted_env: ?*Env, heap: *Heap) Value {
    var env = Rooted(Env).init(heap, unrooted_env);
    var expr = expr_arg;

    tail: while (true) {
        switch (expr.kind) {
            .Func => {
                const func = @fieldParentPtr(Func, "expr", expr);
                if (func.jit_code == null)
                    jit_candidates.put(func, {}) catch std.process.exit(1);
                var closure = Closure.init(heap, &env, func);
                return Value.init(&closure.obj);
            },
            .Var => {
                const var_ = @fieldParentPtr(Var, "expr", expr);
                return Env.lookup(env.get().?, var_.depth);
            },
            .Prim => {
                const prim = @fieldParentPtr(Prim, "expr", expr);
                const lhs = eval_(prim.lhs, env.get(), heap);
                if (!lhs.isSmi())
                    signalError("primcall expected integer lhs, got", lhs.kindName());
                const rhs = eval_(prim.rhs, env.get(), heap);
                if (!rhs.isSmi())
                    signalError("primcall expected integer rhs, got", rhs.kindName());
                return evalPrimcall(prim.op, lhs.getSmi(), rhs.getSmi());
            },
            .Literal => {
                const literal = @fieldParentPtr(Literal, "expr", expr);
                return Value.initVal(literal.val);
            },
            .Call => {
                const call = @fieldParentPtr(Call, "expr", expr);
                const func = Rooted(Value).init(heap, eval_(call.func, env.get(), heap));
                var func_val = func.get();
                if (!func_val.isClosure())
                    signalError("call expected closure, got", func_val.kindName());
                var arg = Rooted(Value).init(heap, eval_(call.arg, env.get(), heap));
                const closure = func_val.asClosure();
                var closure_env = Rooted(Env).init(heap, closure.env);
                var call_env = Env.init(heap, &closure_env, &arg);
                if (closure.func.jit_code) |jit_code| {
                    const f = @ptrCast(*const JitFunction, jit_code);
                    return f(&call_env, heap);
                } else {
                    expr = closure.func.body;
                    env.set(&call_env);
                    continue :tail;
                }
            },
            .LetRec => {
                const letrec = @fieldParentPtr(LetRec, "expr", expr);
                {
                    var filler = Rooted(Value).init(heap, Value.initVal(0));
                    var letrec_env = Env.init(heap, &env, &filler);
                    env.set(&letrec_env);
                }
                const arg = eval_(letrec.arg, env.get(), heap);
                env.get().?.val = arg;
                expr = letrec.body;
                continue :tail;
            },
            .If => {
                const if_ = @fieldParentPtr(If, "expr", expr);
                const test_ = eval_(if_.test_, env.get(), heap);
                if (!test_.isSmi())
                    signalError("if expected integer, got", test_.kindName());
                expr = if (test_.getSmi() != 0) if_.consequent else if_.alternate;
                continue :tail;
            },
            // else => {
            //     signalError("unexpected expr kind", null);
            //     return Value.init(null);
            // },
        }
    }
}

export fn eval(expr: *Expr, heap_size: usize) void {
    var heap = Heap.init(heap_size);
    const res = eval_(expr, null, &heap);
    std.log.info("result: {}", .{res.getSmi()});
}

export fn allocateBytes(len: usize) *anyopaque {
    return (gpa.allocator().alloc(u8, len) catch std.process.exit(1)).ptr;
}
