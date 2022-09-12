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

pub fn log(
    comptime message_level: std.log.Level,
    comptime scope: @Type(.EnumLiteral),
    comptime format: []const u8,
    args: anytype,
) void {
    _ = message_level;
    _ = scope;
    const string = std.fmt.allocPrint(gpa.allocator(), format, args) catch memError();
    defer gpa.allocator().free(string);
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

fn memError() noreturn {
    std.process.exit(255);
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
    body: *Expr, // unique_ptr
    jit_code: ?*anyopaque, // void *

    fn offsetOfBody() usize {
        return @offsetOf(Func, "body");
    }

    fn offsetOfJitCode() usize {
        return @offsetOf(Func, "jit_code");
    }

    fn init(body: *Expr) Func {
        return .{
            .expr = Expr.init(Expr.Kind.Func),
            .body = body,
            .jit_code = null,
        };
    }

    fn initAlloc(body: *Expr) *Func {
        var self = gpa.allocator().create(Func) catch memError();
        self.* = init(body);
        return self;
    }
};

const LetRec = struct {
    const arg_count: u32 = 1;

    expr: Expr, // base class
    arg: *Expr, // unique_ptr
    body: *Expr, // unique_ptr

    fn init(arg: *Expr, body: *Expr) LetRec {
        return .{
            .expr = Expr.init(Expr.Kind.LetRec),
            .arg = arg,
            .body = body,
        };
    }

    fn initAlloc(arg: *Expr, body: *Expr) *LetRec {
        var self = gpa.allocator().create(LetRec) catch memError();
        self.* = init(arg, body);
        return self;
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

    fn initAlloc(depth: u32) *Var {
        var self = gpa.allocator().create(Var) catch memError();
        self.* = init(depth);
        return self;
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
    lhs: *Expr, // unique_ptr
    rhs: *Expr, // unique_ptr

    fn init(op: Op, lhs: *Expr, rhs: *Expr) Prim {
        return .{
            .expr = Expr.init(Expr.Kind.Prim),
            .op = op,
            .lhs = lhs,
            .rhs = rhs,
        };
    }

    fn initAlloc(op: Op, lhs: *Expr, rhs: *Expr) *Prim {
        var self = gpa.allocator().create(Prim) catch memError();
        self.* = init(op, lhs, rhs);
        return self;
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

    fn initAlloc(val: i32) *Literal {
        var self = gpa.allocator().create(Literal) catch memError();
        self.* = init(val);
        return self;
    }
};

const Call = struct {
    expr: Expr, // base class
    func: *Expr, // unique_ptr
    arg: *Expr, // unique_ptr

    fn init(func: *Expr, arg: *Expr) Call {
        return .{
            .expr = Expr.init(Expr.Kind.Call),
            .func = func,
            .arg = arg,
        };
    }

    fn initAlloc(func: *Expr, arg: *Expr) *Call {
        var self = gpa.allocator().create(Call) catch memError();
        self.* = init(func, arg);
        return self;
    }
};

const If = struct {
    expr: Expr, // base class
    test_: *Expr, // unique_ptr
    consequent: *Expr, // unique_ptr
    alternate: *Expr, // unique_ptr

    fn init(test_: *Expr, consequent: *Expr, alternate: *Expr) If {
        return .{
            .expr = Expr.init(Expr.Kind.If),
            .test_ = test_,
            .consequent = consequent,
            .alternate = alternate,
        };
    }

    fn initAlloc(test_: *Expr, consequent: *Expr, alternate: *Expr) *If {
        var self = gpa.allocator().create(If) catch memError();
        self.* = init(test_, consequent, alternate);
        return self;
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

    fn parse(self: *Parser) *const Expr {
        const e = self.parseOne();
        self.skipWhitespace();
        if (!self.eof())
            self.err("expected end of input after expression");
        return e;
    }

    fn pushBound(self: *Parser, id: []const u8) void {
        self.bound_vars.append(id) catch memError();
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

    fn parsePrim(self: *Parser, op: Prim.Op) *Expr {
        const lhs = self.parseOne();
        const rhs = self.parseOne();
        return &Prim.initAlloc(op, lhs, rhs).expr;
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

    fn parseOne(self: *Parser) *Expr {
        self.skipWhitespace();
        if (self.eof())
            self.err("unexpected end of input");
        if (self.matchChar('(')) {
            self.skipWhitespace();

            var ret: *Expr = undefined;
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
                ret = &Func.initAlloc(body).expr;
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
                ret = &LetRec.initAlloc(arg, body).expr;
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
                ret = &If.initAlloc(test_, consequent, alternate).expr;
            } else {
                // Otherwise it's a call.
                const func = self.parseOne();
                const arg = self.parseOne();
                ret = &Call.initAlloc(func, arg).expr;
            }
            self.skipWhitespace();
            if (!self.matchChar(')'))
                self.err("expected close parenthesis");
            return ret;
        } else if (self.startsIdentifier()) {
            return &Var.initAlloc(self.lookupBound(self.takeIdentifier())).expr;
        } else if (isNumeric(self.peek())) {
            return &Literal.initAlloc(self.parseInt32()).expr;
        }
        self.err("unexpected input");
    }
};

export fn parse(str: [*:0]const u8) *const Expr {
    std.log.info("parser input: {s}", .{str});
    var parser = Parser.init(arena.allocator(), str);
    const result = parser.parse();
    std.log.info("parser output: {}", .{result});
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
        heap.roots.append(v) catch memError();
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
    prev: ?*Env,
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
            env_ = env_.prev.?;
        }
        return env_.val;
    }

    fn init(heap: *Heap, prev: *Rooted(Env), val: *Rooted(Value)) Env {
        return .{
            .obj = HeapObject.init(heap, @sizeOf(Env), .Env),
            .prev = prev.get(),
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
var jit_candidates = std.AutoHashMap(*Func, void).init(gpa.allocator());

const JitFunction = fn (*Env, *Heap) Value;

fn eval_(expr_arg: *Expr, unrooted_env: ?*Env, heap: *Heap) Value {
    var env = Rooted(Env).init(heap, unrooted_env);
    var expr = expr_arg;

    tail: while (true) {
        switch (expr.kind) {
            .Func => {
                const func = @fieldParentPtr(Func, "expr", expr);
                if (func.jit_code == null)
                    jit_candidates.put(func, {}) catch memError();
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

const WasmSimpleBlockType = enum(u8) {
    Void = 0x40, // SLEB128(-0x40)
};

const WasmValType = enum(u8) {
    I32 = 0x7f, // SLEB128(-0x01)
    I64 = 0x7e, // SLEB128(-0x02)
    F32 = 0x7d, // SLEB128(-0x03)
    F64 = 0x7c, // SLEB128(-0x04)
    FuncRef = 0x70, // SLEB128(-0x10)
};

// const WasmResultType = std.ArrayList(WasmValType);
const WasmResultType = []const WasmValType;

const WasmFuncType = struct {
    params: WasmResultType,
    results: WasmResultType,
};

const WasmFunc = struct {
    type_idx: usize,
    locals: std.ArrayList(WasmValType),
    code: std.ArrayList(u8),
};

const WasmWriter = struct {
    code: std.ArrayList(u8) = std.ArrayList(u8).init(gpa.allocator()),

    fn finish(self: WasmWriter) std.ArrayList(u8) {
        return self.code;
    }

    fn emit(self: *WasmWriter, byte: u8) void {
        self.code.append(byte) catch memError();
    }

    fn emitVarU32(self: *WasmWriter, i_: u32) void {
        var i = i_;
        while (true) {
            var byte = @intCast(u8, i & 0x7f);
            i >>= 7;
            if (i != 0)
                byte |= 0x80;
            self.emit(byte);
            if (i == 0) break;
        }
    }

    fn emitVarI32(self: *WasmWriter, i_: i32) void {
        var i = i_;
        var done: bool = undefined;
        while (true) {
            var byte = @intCast(u8, i & 0x7f);
            i >>= 7;
            done = ((i == 0) and (byte & 0x40) == 0) or ((i == -1) and (byte & 0x40) != 0);
            if (!done)
                byte |= 0x80;
            self.emit(byte);
            if (done) break;
        }
    }

    fn emitPatchableVarU32(self: *WasmWriter) usize {
        const offset = self.code.items.len;
        self.emitVarU32(std.math.maxInt(u32));
        return offset;
    }

    fn emitPatchableVarI32(self: *WasmWriter) usize {
        const offset = self.code.items.len;
        self.emitVarI32(std.math.maxInt(i32));
        return offset;
    }

    fn patchVarI32(self: *WasmWriter, offset: usize, val_: i32) void {
        var val = val_;
        var i: usize = 0;
        while (i < 5) : ({
            i += 1;
            val >>= 7;
        }) {
            var byte = @intCast(u8, val & 0x7f);
            if (i < 4)
                byte |= 0x80;
            self.code.items[offset + i] = byte;
        }
    }

    fn patchVarU32(self: *WasmWriter, offset: usize, val_: u32) void {
        var val = val_;
        var i: usize = 0;
        while (i < 5) : ({
            i += 1;
            val >>= 7;
        }) {
            var byte = @intCast(u8, val & 0x7f);
            if (i < 4)
                byte |= 0x80;
            self.code.items[offset + i] = byte;
        }
    }

    fn emitValType(self: *WasmWriter, t: WasmValType) void {
        self.emit(@enumToInt(t));
    }
};

const WasmAssembler = struct {
    writer: WasmWriter = WasmWriter{},

    const Op = enum(u8) {
        Unreachable = 0x00,
        Nop = 0x01,
        Block = 0x02,
        Loop = 0x03,
        If = 0x04,
        Else = 0x05,
        End = 0x0b,
        Br = 0x0c,
        BrIf = 0x0d,
        Return = 0x0f,

        // Call operators
        Call = 0x10,
        CallIndirect = 0x11,

        // Parametric operators
        Drop = 0x1a,

        // Variable access
        LocalGet = 0x20,
        LocalSet = 0x21,
        LocalTee = 0x22,

        // Memory-related operators
        I32Load = 0x28,
        I32Store = 0x36,

        // Constants
        I32Const = 0x41,

        // Comparison operators
        I32Eqz = 0x45,
        I32Eq = 0x46,
        I32Ne = 0x47,
        I32LtS = 0x48,
        I32LtU = 0x49,

        // Numeric operators
        I32Add = 0x6a,
        I32Sub = 0x6b,
        I32Mul = 0x6c,
        I32And = 0x71,
        I32Or = 0x72,
        I32Xor = 0x73,
        I32Shl = 0x74,
        I32ShrS = 0x75,
        I32ShrU = 0x76,

        RefNull = 0xd0,

        MiscPrefix = 0xfc,
    };

    const MiscOp = enum(u8) {
        TableGrow = 0x0f,
        TableInit = 0x0c,
    };

    fn emitOp(self: *WasmAssembler, op: Op) void {
        self.writer.emit(@enumToInt(op));
    }

    fn emitPatchableI32Const(self: *WasmAssembler) void {
        self.emitOp(.I32Const);
        return self.writer.emitPatchableVarI32();
    }

    fn emitI32Const(self: *WasmAssembler, val: i32) void {
        self.emitOp(.I32Const);
        return self.writer.emitVarI32(val);
    }

    fn emitMemArg(self: *WasmAssembler, align_: i32, offset: u32) void {
        self.writer.emitVarU32(@intCast(u32, align_));
        self.writer.emitVarU32(@intCast(u32, offset));
    }

    const Int32SizeLog2 = 2;

    fn emitI32Load(self: *WasmAssembler, offset: ?u32) void {
        self.emitOp(.I32Load);
        self.emitMemArg(Int32SizeLog2, offset orelse 0);
    }

    fn emitI32Store(self: *WasmAssembler, offset: ?u32) void {
        self.emitOp(.I32Store);
        self.emitMemArg(Int32SizeLog2, offset orelse 0);
    }

    fn emitLocalGet(self: *WasmAssembler, idx: u32) void {
        self.emitOp(.LocalGet);
        self.writer.emitVarU32(idx);
    }

    fn emitLocalSet(self: *WasmAssembler, idx: u32) void {
        self.emitOp(.LocalSet);
        self.writer.emitVarU32(idx);
    }

    fn emitLocalTee(self: *WasmAssembler, idx: u32) void {
        self.emitOp(.LocalTee);
        self.writer.emitVarU32(idx);
    }

    fn emitI32Eqz(self: *WasmAssembler) void {
        self.emitOp(.I32Eqz);
    }

    fn emitI32Eq(self: *WasmAssembler) void {
        self.emitOp(.I32Eq);
    }

    fn emitI32Ne(self: *WasmAssembler) void {
        self.emitOp(.I32Ne);
    }

    fn emitI32LtS(self: *WasmAssembler) void {
        self.emitOp(.I32LtS);
    }

    fn emitI32LtU(self: *WasmAssembler) void {
        self.emitOp(.I32LtU);
    }

    fn emitI32Add(self: *WasmAssembler) void {
        self.emitOp(.I32Add);
    }

    fn emitI32Sub(self: *WasmAssembler) void {
        self.emitOp(.I32Sub);
    }

    fn emitI32Mul(self: *WasmAssembler) void {
        self.emitOp(.I32Mul);
    }

    fn emitI32And(self: *WasmAssembler) void {
        self.emitOp(.I32And);
    }

    fn emitI32Or(self: *WasmAssembler) void {
        self.emitOp(.I32Or);
    }

    fn emitI32Xor(self: *WasmAssembler) void {
        self.emitOp(.I32Xor);
    }

    fn emitI32Shl(self: *WasmAssembler) void {
        self.emitOp(.I32Shl);
    }

    fn emitI32ShrS(self: *WasmAssembler) void {
        self.emitOp(.I32ShrS);
    }

    fn emitI32ShrU(self: *WasmAssembler) void {
        self.emitOp(.I32ShrU);
    }

    fn emitCallIndirect(self: *WasmAssembler, callee_type: u32, table: ?u32) void {
        self.emitOp(.CallIndirect);
        self.writer.emitVarU32(callee_type);
        self.writer.emitVarU32(table orelse 0);
    }

    fn emitRefNull(self: *WasmAssembler, type_: WasmValType) void {
        self.emitOp(.RefNull);
        self.writer.emitValType(type_);
    }

    fn emitMiscOp(self: *WasmAssembler, op: MiscOp) void {
        self.emitOp(.MiscPrefix);
        self.writer.emit(@enumToInt(op));
    }

    fn emitTableGrow(self: *WasmAssembler, idx: u32) void {
        self.emitMiscOp(.TableGrow);
        self.writer.emitVarU32(idx);
    }

    fn emitTableInit(self: *WasmAssembler, dst: u32, src: u32) void {
        self.emitMiscOp(.TableInit);
        self.writer.emitVarU32(dst);
        self.writer.emitVarU32(src);
    }

    fn emitBlock(self: *WasmAssembler, block_type: ?WasmValType) void {
        self.emitOp(.Block);
        self.writer.emit(if (block_type) |t| @enumToInt(t) else @enumToInt(WasmSimpleBlockType.Void));
    }

    fn emitEnd(self: *WasmAssembler) void {
        self.emitOp(.End);
    }

    fn emitBr(self: *WasmAssembler, offset: u32) void {
        self.emitOp(.Br);
        self.writer.emitVarU32(offset);
    }

    fn emitBrIf(self: *WasmAssembler, offset: u32) void {
        self.emitOp(.BrIf);
        self.writer.emitVarU32(offset);
    }

    fn emitUnreachable(self: *WasmAssembler) void {
        self.emitOp(.Unreachable);
    }

    fn emitReturn(self: *WasmAssembler) void {
        self.emitOp(.Return);
    }
};

const WasmModuleWriter = struct {
    writer: WasmWriter = WasmWriter{},

    const SectionId = enum(u8) {
        Custom = 0,
        Type = 1,
        Import = 2,
        Function = 3,
        Table = 4,
        Memory = 5,
        Global = 6,
        Export = 7,
        Start = 8,
        Elem = 9,
        Code = 10,
        Data = 11,
        DataCount = 12,
    };

    const DefinitionKind = enum(u8) {
        Function = 0x00,
        Table = 0x01,
        Memory = 0x02,
        Global = 0x03,
    };

    const LimitsFlags = enum(u8) {
        Default = 0x0,
        HasMaximum = 0x1,
        IsShared = 0x2,
        IsI64 = 0x4,
    };

    const ElemSegmentKind = enum(u8) {
        Active = 0x0,
        Passive = 0x1,
    };

    fn emitMagic(self: *WasmModuleWriter) void {
        self.writer.emit(0x00);
        self.writer.emit(0x61);
        self.writer.emit(0x73);
        self.writer.emit(0x6d);
    }

    fn emitVersion(self: *WasmModuleWriter) void {
        self.writer.emit(0x01);
        self.writer.emit(0x00);
        self.writer.emit(0x00);
        self.writer.emit(0x00);
    }

    // fn emitResultType(self: *WasmModuleWriter, type_: *const WasmResultType) void {
    fn emitResultType(self: *WasmModuleWriter, type_: []const WasmValType) void {
        self.writer.emitVarU32(type_.len);
        for (type_) |t| {
            self.writer.emitValType(t);
        }
    }

    fn emitSectionId(self: *WasmModuleWriter, id: SectionId) void {
        self.writer.emit(@enumToInt(id));
    }

    fn emitTypeSection(self: *WasmModuleWriter, types: *const std.ArrayList(WasmFuncType)) void {
        self.emitSectionId(.Type);
        const patch_loc = self.writer.emitPatchableVarU32();
        const start = self.writer.code.items.len;
        self.writer.emitVarU32(types.items.len);
        for (types.items) |type_| {
            self.writer.emit(0x60);
            self.emitResultType(type_.params);
            self.emitResultType(type_.results);
        }
        self.writer.patchVarU32(patch_loc, self.writer.code.items.len - start);
    }

    fn emitName(self: *WasmModuleWriter, name: []const u8) void {
        self.writer.emitVarU32(name.len);
        for (name) |char|
            self.writer.emit(char);
    }

    fn emitImportSection(self: *WasmModuleWriter) void {
        self.emitSectionId(.Import);
        const patch_loc = self.writer.emitPatchableVarU32();
        const start = self.writer.code.items.len;
        self.writer.emitVarU32(2);
        self.emitName("env");
        self.emitName("memory");
        self.writer.emit(@enumToInt(DefinitionKind.Memory));
        self.writer.emit(@enumToInt(LimitsFlags.Default));
        self.writer.emitVarU32(0);
        self.emitName("env");
        self.emitName("__indirect_function_table");
        self.writer.emit(@enumToInt(DefinitionKind.Table));
        self.writer.emitValType(.FuncRef);
        self.writer.emit(@enumToInt(LimitsFlags.Default));
        self.writer.emitVarU32(0);
        self.writer.patchVarU32(patch_loc, self.writer.code.items.len - start);
    }

    fn emitFunctionSection(self: *WasmModuleWriter, funcs: *const std.ArrayList(WasmFunc)) void {
        self.emitSectionId(.Function);
        const patch_loc = self.writer.emitPatchableVarU32();
        const start = self.writer.code.items.len;
        self.writer.emitVarU32(funcs.items.len);
        for (funcs.items) |func|
            self.writer.emitVarU32(func.type_idx);
        self.writer.patchVarU32(patch_loc, self.writer.code.items.len - start);
    }

    fn emitElementSection(self: *WasmModuleWriter, indirect_functions: *const std.ArrayList(u32)) void {
        if (indirect_functions.items.len == 0) return;
        self.emitSectionId(.Elem);
        const patch_loc = self.writer.emitPatchableVarU32();
        const start = self.writer.code.items.len;
        self.writer.emitVarU32(1);
        self.writer.emit(@enumToInt(ElemSegmentKind.Passive));
        self.writer.emit(0x00);
        self.writer.emitVarU32(indirect_functions.items.len);
        for (indirect_functions.items) |idx|
            self.writer.emitVarU32(idx);
        self.writer.patchVarU32(patch_loc, self.writer.code.items.len - start);
    }

    fn emitStartSection(self: *WasmModuleWriter, start_function: u32) void {
        self.emitSectionId(.Start);
        const patch_loc = self.writer.emitPatchableVarU32();
        const start = self.writer.code.items.len;
        self.writer.emitVarU32(start_function);
        self.writer.patchVarU32(patch_loc, self.writer.code.items.len - start);
    }

    fn encodeLocals(locals: *const std.ArrayList(WasmValType)) std.ArrayList(u8) {
        var runs: u32 = 0;
        {
            var local: usize = 0;
            while (local < locals.items.len) : (runs += 1) {
                const t = locals.items[local];
                local += 1;
                while (local < locals.items.len and locals.items[local] == t) : (local += 1) {}
            }
        }
        var writer = WasmWriter{};
        writer.emitVarU32(runs);
        {
            var local: usize = 0;
            while (local < locals.items.len) {
                const t = locals.items[local];
                local += 1;
                var count: u32 = 1;
                while (local < locals.items.len and locals.items[local] == t) {
                    count += 1;
                    local += 1;
                }
                writer.emitVarU32(count);
                writer.emitValType(t);
            }
        }
        return writer.finish();
    }

    fn emitCodeSection(self: *WasmModuleWriter, funcs: *const std.ArrayList(WasmFunc)) void {
        self.emitSectionId(.Code);
        const patch_loc = self.writer.emitPatchableVarU32();
        const start = self.writer.code.items.len;
        self.writer.emitVarU32(funcs.items.len);
        for (funcs.items) |func| {
            const locals = encodeLocals(&func.locals);
            self.writer.emitVarU32(locals.items.len + func.code.items.len);
            self.writer.code.appendSlice(locals.items) catch memError();
            self.writer.code.appendSlice(func.code.items) catch memError();
        }
        self.writer.patchVarU32(patch_loc, self.writer.code.items.len - start);
    }
};

const WasmModuleBuilder = struct {
    types: std.ArrayList(WasmFuncType) = std.ArrayList(WasmFuncType).init(gpa.allocator()),
    functions: std.ArrayList(WasmFunc) = std.ArrayList(WasmFunc).init(gpa.allocator()),
    indirect_function_table: std.ArrayList(u32) = std.ArrayList(u32).init(gpa.allocator()),
    start_function: u32 = @bitCast(u32, @as(i32, -1)),

    fn internFuncTypeSlices(self: *WasmModuleBuilder, params: []const WasmValType, results: []const WasmValType) usize {
        var p = WasmResultType.initCapacity(gpa.allocator(), params.len) catch memError();
        var r = WasmResultType.initCapacity(gpa.allocator(), results.len) catch memError();
        p.appendSliceAssumeCapacity(params);
        r.appendSliceAssumeCapacity(results);
        return self.internFuncType(&p, &r);
    }
    // fn internFuncType(self: *WasmModuleBuilder, params: []WasmResultType, results: []WasmResultType) usize {
    fn internFuncType(self: *WasmModuleBuilder, params: []const WasmValType, results: []const WasmValType) usize {
        var i: usize = 0;
        while (i < self.types.items.len) : (i += 1) {
            if (self.types.items[i].params.len != params.len)
                continue;
            if (self.types.items[i].results.len != results.len)
                continue;
            var same = true;
            var j: usize = 0;
            while (j < params.len) : (j += 1) {
                if (self.types.items[i].params[i] != params[i])
                    same = false;
            }
            j = 0;
            while (j < results.len) : (j += 1) {
                if (self.types.items[i].results[i] != results[i])
                    same = false;
            }
            if (same)
                return i;
        }
        // self.types.append(WasmFuncType{ .params = params.*, .results = results.* }) catch memError();
        self.types.append(WasmFuncType{ .params = params, .results = results }) catch memError();
        return self.types.items.len - 1;
    }

    fn addFunction(self: *WasmModuleBuilder, type_: u32, locals: std.ArrayList(WasmValType), code: std.ArrayList(u8)) usize {
        self.functions.append(WasmFunc{ .type_idx = type_, .locals = locals, .code = code }) catch memError();
        return self.functions.items.len - 1;
    }

    fn addIndirectFunction(self: *WasmModuleBuilder, idx: u32) void {
        self.indirect_function_table.append(idx) catch memError();
    }

    fn recordStartFunction(self: *WasmModuleBuilder, idx: u32) void {
        self.start_function = idx;
    }

    fn finish(self: WasmModuleBuilder) std.ArrayList(u8) {
        var writer = WasmModuleWriter{};
        writer.emitMagic();
        writer.emitVersion();
        writer.emitTypeSection(&self.types);
        writer.emitImportSection();
        writer.emitFunctionSection(&self.functions);
        if (self.start_function != @bitCast(u32, @as(i32, -1)))
            writer.emitStartSection(self.start_function);
        writer.emitElementSection(&self.indirect_function_table);
        writer.emitCodeSection(&self.functions);
        return writer.writer.finish();
    }
};

const VMCall = struct {
    fn Allocate(heap: *Heap, bytes: usize) *anyopaque {
        return heap.allocate(bytes);
    }

    fn PushRoot(v: Value, heap: *Heap) usize {
        const ret = Heap.pushRoot(heap, v);
        return ret;
    }

    fn GetRoot(heap: *Heap, idx: usize) Value {
        return Heap.getRoot(heap, idx);
    }

    fn PopRoots(heap: *Heap, n_: usize) void {
        var n = n_;
        while (n > 0) {
            n -= 1;
            Heap.popRoot(heap);
        }
    }

    fn Error(msg: []const u8, what: []const u8) void {
        std.log.err("Error({s}, {s})\n", .{ msg, what });
        signalError(msg, what);
    }

    fn Debug(v: usize, what: []const u8) usize {
        std.log.err("Debug({s}, {})\n", .{ what, v });
        return v;
    }

    fn Eval(expr: *Expr, env: *Env, heap: *Heap) Value {
        std.log.err("Eval({}, {}, {})\n", .{ expr, env, heap });
        return eval_(expr, env, heap);
    }
};

const VMCallTypes = struct {
    initialized: bool = false,
    Allocate: u32 = undefined,
    PushRoot: u32 = undefined,
    GetRoot: u32 = undefined,
    PopRoots: u32 = undefined,
    Error: u32 = undefined,
    Debug: u32 = undefined,
    Eval: u32 = undefined,
    JitCall: u32 = undefined,
    StartFunction: u32 = undefined,
};

const WasmMacroAssembler = struct {
    assembler: WasmAssembler = WasmAssembler{},

    module_builder: WasmModuleBuilder = WasmModuleBuilder{},
    relocs: std.ArrayList(*anyopaque) = std.ArrayList(*anyopaque).init(gpa.allocator()),
    vm_call_types: VMCallTypes = VMCallTypes{},
    max_roots: usize = undefined,
    current_root_count: usize = undefined,
    current_active_locals: usize = undefined,
    locals: std.ArrayList(WasmValType) = std.ArrayList(WasmValType).init(gpa.allocator()),

    const UnrootedEnvLocalIdx = 0;
    const HeapLocalIdx = 1;
    const ParamCount = 2;

    fn acquireLocal(self: *WasmMacroAssembler, type_opt: ?WasmValType) usize {
        const type_ = type_opt orelse .I32;
        var i: usize = self.current_active_locals;
        while (i < self.locals.items.len) : (i += 1) {
            if (self.locals.items[i] == type_) {
                const idx = ParamCount + i;
                self.current_active_locals = i + 1;
                return idx;
            }
        }
        self.locals.append(type_) catch memError();
        self.current_active_locals = ParamCount + self.locals.items.len;
        return self.current_active_locals - 1;
    }

    fn releaseLocal(self: *WasmMacroAssembler) void {
        self.current_active_locals -= 1;
    }

    fn releaseLocals(self: *WasmMacroAssembler, n_: usize) void {
        var n = n_;
        while (n > 0) {
            n -= 1;
            self.releaseLocal();
        }
    }

    fn emitLoadPointer(self: *WasmMacroAssembler, offset: ?usize) void {
        self.assembler.emitI32Load(offset orelse 0);
    }

    fn emitStorePointer(self: *WasmMacroAssembler, offset: ?usize) void {
        self.assembler.emitI32Store(offset orelse 0);
    }

    fn emitUnrootedEnv(self: *WasmMacroAssembler) void {
        self.assembler.emitLocalGet(UnrootedEnvLocalIdx);
    }

    fn emitHeap(self: *WasmMacroAssembler) void {
        self.assembler.emitLocalGet(HeapLocalIdx);
    }

    fn emitVMCall(self: *WasmMacroAssembler, comptime T: type, f: T, type_: u32) void {
        self.assembler.emitI32Const(@intCast(i32, @ptrToInt(f)));
        self.assembler.emitCallIndirect(type_, null);
    }

    fn emitAllocate(self: *WasmMacroAssembler, comptime T: type) void {
        const bytes = @sizeOf(T);
        self.emitHeap();
        self.assembler.emitI32Const(bytes);
        self.emitVMCall(*const @TypeOf(VMCall.Allocate), &VMCall.Allocate, self.vm_call_types.Allocate);
    }

    fn emitStoreGCRoot(self: *WasmMacroAssembler) u32 {
        self.current_root_count += 1;
        if (self.max_roots < self.current_root_count)
            self.max_roots = self.current_root_count;
        const local = self.acquireLocal(null);
        self.assembler.emitLocalTee(local);
        self.emitHeap();
        self.emitVMCall(*const @TypeOf(VMCall.PushRoot), VMCall.PushRoot, self.vm_call_types.PushRoot);
        self.assembler.emitLocalSet(local);
        return local;
    }

    fn emitLoadGCRoot(self: *WasmMacroAssembler, local: u32) void {
        self.emitHeap();
        self.assembler.emitLocalGet(local);
        self.emitVMCall(*const @TypeOf(VMCall.GetRoot), VMCall.GetRoot, self.vm_call_types.GetRoot);
    }

    fn emitPopGCRootsAndReleaseLocals(self: *WasmMacroAssembler, n: usize) void {
        self.current_root_count -= n;
        self.releaseLocals(n);
        self.emitHeap();
        self.assembler.emitI32Const(@intCast(i32, n));
        self.emitVMCall(*const @TypeOf(VMCall.PopRoots), VMCall.PopRoots, self.vm_call_types.PopRoots);
    }

    fn emitHeapObjectInitTag(self: *WasmMacroAssembler, kind: HeapObject.Kind) void {
        var val = @enumToInt(kind);
        val <<= HeapObject.NotForwardedBits;
        val |= HeapObject.NotForwardedBit;
        self.assembler.emitI32Const(@intCast(i32, val));
        self.emitStorePointer(HeapObject.offsetOfTag());
    }

    fn emitPushConstantPointer(self: *WasmMacroAssembler, ptr: *const anyopaque) void {
        self.assembler.emitI32Const(@intCast(i32, @ptrToInt(ptr)));
    }

    fn emitAssertionFailure(self: *WasmMacroAssembler, msg: []const u8, what: []const u8) void {
        self.emitPushConstantPointer(msg.ptr);
        self.emitPushConstantPointer(what.ptr);
        self.emitVMCall(*const @TypeOf(VMCall.Error), VMCall.Error, self.vm_call_types.Error);
        self.assembler.emitUnreachable();
    }

    fn emitDebug(self: *WasmMacroAssembler, what: []const u8) void {
        self.emitPushConstantPointer(what.ptr);
        self.emitVMCall(*const @TypeOf(VMCall.Debug), VMCall.Debug, self.vm_call_types.Debug);
    }

    fn emitCheckSmi(self: *WasmMacroAssembler, local_idx: usize, what: []const u8) void {
        self.assembler.emitBlock(null);
        self.assembler.emitLocalGet(local_idx);
        self.assembler.emitI32Const(Value.TagMask);
        self.assembler.emitI32And();
        self.assembler.emitI32Const(Value.SmiTag);
        self.assembler.emitI32Eq();
        self.assembler.emitBrIf(0);
        self.emitAssertionFailure("expected an integer", what);
        self.assembler.emitEnd();
    }

    fn emitValueToSmi(self: *WasmMacroAssembler) void {
        self.assembler.emitI32Const(Value.TagBits);
        self.assembler.emitI32ShrS();
    }

    fn emitSmiToValue(self: *WasmMacroAssembler) void {
        self.assembler.emitI32Const(Value.TagBits);
        self.assembler.emitI32Shl();
        self.assembler.emitI32Const(Value.SmiTag);
        self.assembler.emitI32Or();
    }

    fn emitCheckHeapObject(self: *WasmMacroAssembler, local_idx: usize, kind: HeapObject.Kind, what: []const u8) void {
        self.assembler.emitBlock(null);
        self.assembler.emitLocalGet(local_idx);
        self.assembler.emitI32Const(Value.TagMask);
        self.assembler.emitI32And();
        self.assembler.emitI32Eqz();
        self.assembler.emitBrIf(0);
        self.emitAssertionFailure("expected an heap object", what);
        self.assembler.emitEnd();

        self.assembler.emitBlock(null);
        self.assembler.emitLocalGet(local_idx);
        self.emitLoadPointer(null);
        self.assembler.emitI32Const(HeapObject.NotForwardedBits);
        self.assembler.emitI32ShrU();
        self.assembler.emitI32Const(@intCast(i32, @enumToInt(kind)));
        self.assembler.emitI32Eq();
        self.assembler.emitBrIf(0);
        self.emitAssertionFailure("expected a different heap object kind", what);
        self.assembler.emitEnd();
    }

    fn emitValueToHeapObject(self: *WasmMacroAssembler) void {
        _ = self;
    }

    fn emitHeapObjectToValue(self: *WasmMacroAssembler) void {
        _ = self;
    }

    fn initializeVMCallTypes(self: *WasmMacroAssembler) void {
        const arr = [_]WasmValType{ .I32, .I32, .I32 };
        const Call_0_0 = self.module_builder.internFuncType(arr[0..0], arr[0..0]);
        const Call_2_0 = self.module_builder.internFuncType(arr[0..2], arr[0..0]);
        const Call_2_1 = self.module_builder.internFuncType(arr[0..2], arr[0..1]);
        const Call_3_1 = self.module_builder.internFuncType(arr[0..], arr[0..1]);
        self.vm_call_types = .{
            .Allocate = Call_2_1,
            .PushRoot = Call_2_1,
            .GetRoot = Call_2_1,
            .PopRoots = Call_2_0,
            .Error = Call_2_0,
            .Debug = Call_2_1,
            .Eval = Call_3_1,
            .JitCall = Call_2_1,
            .StartFunction = Call_0_0,
            .initialized = true,
        };
    }

    fn beginFunction(self: *WasmMacroAssembler) void {
        if (!self.vm_call_types.initialized)
            self.initializeVMCallTypes();

        self.current_active_locals = 0;
        self.current_root_count = 0;
        self.max_roots = 0;
        self.locals.clearAndFree();
        self.assembler.writer.code.clearAndFree();
    }

    fn endFunction(self: *WasmMacroAssembler) u32 {
        self.assembler.emitReturn();
        self.assembler.emitEnd();
        return self.module_builder.addFunction(self.vm_call_types.JitCall, self.locals, self.assembler.writer.finish());
    }

    fn recordRelocation(self: *WasmMacroAssembler, address: *anyopaque, func_idx: u32) void {
        self.module_builder.addIndirectFunction(func_idx);
        self.relocs.append(address) catch memError();
    }

    fn emitRelocations(self: *WasmMacroAssembler) void {
        const count = @intCast(i32, self.relocs.items.len);
        self.beginFunction();
        self.locals.append(.I32) catch memError();
        const base = 0;
        self.assembler.emitRefNull(.FuncRef);
        self.assembler.emitI32Const(count);
        self.assembler.emitTableGrow(0);
        self.assembler.emitLocalSet(base);

        self.assembler.emitLocalGet(base);
        self.assembler.emitI32Const(0);
        self.assembler.emitI32Const(count);
        self.assembler.emitTableInit(0, 0);

        var i: u32 = 0;
        while (i < count) : (i += 1) {
            self.assembler.emitI32Const(@intCast(i32, @ptrToInt(self.relocs.items[i])));
            self.assembler.emitLocalGet(base);
            self.assembler.emitI32Const(@intCast(i32, i));
            self.assembler.emitI32Add();
            self.assembler.emitI32Store(null);
        }

        self.assembler.emitEnd();
        const offset = self.module_builder.addFunction(self.vm_call_types.StartFunction, self.locals, self.assembler.writer.finish());

        self.module_builder.recordStartFunction(offset);
        self.relocs.clearAndFree();
    }

    fn endModule(self: *WasmMacroAssembler) std.ArrayList(u8) {
        self.emitRelocations();
        return self.module_builder.finish();
    }
};

const WasmCompiler = struct {
    masm: WasmMacroAssembler = WasmMacroAssembler{},

    fn compile(self: *WasmCompiler, expr: *Expr, env_root: usize) void {
        switch (expr.kind) {
            .Func => {
                const func = @fieldParentPtr(Func, "expr", expr);
                self.masm.emitAllocate(Closure);
                const local = self.masm.acquireLocal(null);
                self.masm.assembler.emitLocalTee(local);
                self.masm.emitHeapObjectInitTag(.Closure);
                self.masm.assembler.emitLocalGet(local);
                self.masm.emitLoadGCRoot(env_root);
                self.masm.emitStorePointer(Closure.offsetOfEnv());
                self.masm.assembler.emitLocalGet(local);
                self.masm.emitPushConstantPointer(func);
                self.masm.emitStorePointer(Closure.offsetOfFunc());
                self.masm.assembler.emitLocalGet(local);
                self.masm.releaseLocal();
            },
            .Var => {
                const var_ = @fieldParentPtr(Var, "expr", expr);
                self.masm.emitLoadGCRoot(env_root);
                var depth = var_.depth;
                while (depth > 0) {
                    depth -= 1;
                    self.masm.emitLoadPointer(Env.offsetOfPrev());
                }
                self.masm.emitLoadPointer(Env.offsetOfVal());
            },
            .Prim => {
                const prim = @fieldParentPtr(Prim, "expr", expr);
                self.compile(prim.lhs, env_root);
                const lhs = self.masm.acquireLocal(null);
                self.masm.assembler.emitLocalSet(lhs);
                self.masm.emitCheckSmi(lhs, "primcall");
                self.compile(prim.rhs, env_root);
                const rhs = self.masm.acquireLocal(null);
                self.masm.assembler.emitLocalSet(rhs);
                self.masm.emitCheckSmi(rhs, "primcall");

                self.masm.assembler.emitLocalGet(lhs);
                self.masm.emitValueToSmi();
                self.masm.assembler.emitLocalGet(rhs);
                self.masm.emitValueToSmi();
                switch (prim.op) {
                    .Eq => self.masm.assembler.emitI32Eq(),
                    .LessThan => self.masm.assembler.emitI32LtS(),
                    .Add => self.masm.assembler.emitI32Add(),
                    .Sub => self.masm.assembler.emitI32Sub(),
                    .Mul => self.masm.assembler.emitI32Mul(),
                }
                self.masm.emitSmiToValue();
                self.masm.releaseLocals(2);
            },
            .Literal => {
                const literal = @fieldParentPtr(Literal, "expr", expr);
                const v = Value.initVal(literal.val);
                self.masm.assembler.emitI32Const(@intCast(i32, v.bits()));
            },
            .Call => {
                const call = @fieldParentPtr(Call, "expr", expr);
                self.compile(call.func, env_root);

                const unrooted_callee = self.masm.acquireLocal(null);
                const unrooted_env = self.masm.acquireLocal(null);

                self.masm.assembler.emitLocalSet(unrooted_callee);
                self.masm.emitCheckHeapObject(unrooted_callee, .Closure, "call");
                self.masm.assembler.emitLocalGet(unrooted_callee);
                const callee = self.masm.emitStoreGCRoot();
                self.compile(call.arg, env_root);
                const arg = self.masm.emitStoreGCRoot();
                self.masm.emitAllocate(Env);
                // unrooted_callee now invalid.

                self.masm.assembler.emitLocalTee(unrooted_env);
                self.masm.emitHeapObjectInitTag(.Env);
                self.masm.assembler.emitLocalGet(unrooted_env);
                self.masm.emitLoadGCRoot(callee);
                self.masm.emitLoadPointer(Closure.offsetOfEnv());
                self.masm.emitStorePointer(Env.offsetOfPrev());
                self.masm.assembler.emitLocalGet(unrooted_env);
                self.masm.emitLoadGCRoot(arg);
                self.masm.emitStorePointer(Env.offsetOfVal());

                self.masm.emitLoadGCRoot(callee);
                self.masm.assembler.emitLocalSet(unrooted_callee);
                self.masm.emitPopGCRootsAndReleaseLocals(2);
                // Now unrooted_env and unrooted_callee valid, gcroots popped.

                self.masm.assembler.emitBlock(.I32);
                self.masm.assembler.emitBlock(null);
                self.masm.assembler.emitLocalGet(unrooted_callee);
                self.masm.emitLoadPointer(Closure.offsetOfFunc());
                self.masm.emitLoadPointer(Func.offsetOfJitCode());
                // If there is jit code, jump out.
                self.masm.assembler.emitBrIf(0);

                // No jit code?  Call eval.  FIXME: tail calls.
                self.masm.assembler.emitLocalGet(unrooted_callee);
                self.masm.emitLoadPointer(Closure.offsetOfFunc());
                self.masm.emitLoadPointer(Func.offsetOfBody());
                self.masm.assembler.emitLocalGet(unrooted_env);
                self.masm.emitHeap();
                self.masm.emitVMCall(*const @TypeOf(VMCall.Eval), VMCall.Eval, self.masm.vm_call_types.Eval);
                self.masm.assembler.emitBr(1); // Called eval, jump past jit call with result.
                self.masm.assembler.emitEnd();

                // Otherwise if we get here there's JIT code.
                self.masm.assembler.emitLocalGet(unrooted_env);
                self.masm.emitHeap();
                self.masm.assembler.emitLocalGet(unrooted_callee);
                self.masm.emitLoadPointer(Closure.offsetOfFunc());
                self.masm.emitLoadPointer(Func.offsetOfJitCode());
                self.masm.assembler.emitCallIndirect(self.masm.vm_call_types.JitCall, null);
                self.masm.assembler.emitEnd();

                self.masm.releaseLocals(2);
            },
            .LetRec => {
                const letrec = @fieldParentPtr(LetRec, "expr", expr);
                self.masm.emitAllocate(Env);
                {
                    const unrooted_env = self.masm.acquireLocal(null);
                    self.masm.assembler.emitLocalTee(unrooted_env);
                    self.masm.emitHeapObjectInitTag(.Env);
                    self.masm.assembler.emitLocalGet(unrooted_env);
                    self.masm.emitLoadGCRoot(env_root);
                    self.masm.emitStorePointer(Env.offsetOfPrev());
                    self.masm.assembler.emitLocalGet(unrooted_env);
                    self.masm.assembler.emitI32Const(@intCast(i32, Value.initVal(0).bits()));
                    self.masm.emitStorePointer(Env.offsetOfVal());
                    self.masm.assembler.emitLocalGet(unrooted_env);
                    self.masm.releaseLocal();
                }
                const env = self.masm.emitStoreGCRoot();
                self.compile(letrec.arg, env);
                {
                    const unrooted_arg = self.masm.acquireLocal(null);
                    self.masm.assembler.emitLocalSet(unrooted_arg);
                    self.masm.emitLoadGCRoot(env);
                    self.masm.assembler.emitLocalGet(unrooted_arg);
                    self.masm.emitStorePointer(Env.offsetOfVal());
                    self.masm.releaseLocal();
                }

                self.compile(letrec.body, env);
                self.masm.emitPopGCRootsAndReleaseLocals(1);
            },
            .If => {
                const if_ = @fieldParentPtr(If, "expr", expr);
                self.compile(if_.test_, env_root);
                {
                    const test_ = self.masm.acquireLocal(null);
                    self.masm.assembler.emitLocalSet(test_);
                    self.masm.emitCheckSmi(test_, "conditional");
                    self.masm.assembler.emitBlock(.I32);
                    self.masm.assembler.emitBlock(null);
                    self.masm.assembler.emitLocalGet(test_);
                    self.masm.emitValueToSmi();
                    self.masm.releaseLocal();
                }
                self.masm.assembler.emitBrIf(0);
                self.compile(if_.alternate, env_root);
                self.masm.assembler.emitBr(1);
                self.masm.assembler.emitEnd();
                self.compile(if_.consequent, env_root);
                self.masm.assembler.emitEnd();
            },
        }
    }

    fn compileFunction(self: *WasmCompiler, func: *Func) void {
        self.masm.beginFunction();
        self.masm.emitUnrootedEnv();
        const env = self.masm.emitStoreGCRoot();
        self.compile(func.body, env);
        self.masm.emitPopGCRootsAndReleaseLocals(1);
        const offset = self.masm.endFunction();
        self.masm.recordRelocation(&func.jit_code, offset);
    }

    fn finish(self: *WasmCompiler) std.ArrayList(u8) {
        return self.masm.endModule();
    }
};

const WasmModule = struct {
    data: std.ArrayList(u8),
};

export fn jitModule() ?*WasmModule {
    if (jit_candidates.count() == 0)
        return null;

    var comp = WasmCompiler{};
    var it = jit_candidates.keyIterator();
    while (it.next()) |f| {
        comp.compileFunction(f.*);
    }
    jit_candidates.clearAndFree();
    var mod = gpa.allocator().create(WasmModule) catch memError();
    mod.data = comp.finish();
    return mod;
}

export fn moduleData(mod: *WasmModule) [*]u8 {
    return mod.data.items.ptr;
}

export fn moduleSize(mod: *WasmModule) usize {
    return mod.data.items.len;
}

export fn freeModule(mod: *WasmModule) void {
    gpa.allocator().destroy(mod);
}

export fn allocateBytes(len: usize) *anyopaque {
    return (gpa.allocator().alloc(u8, len) catch memError()).ptr;
}
